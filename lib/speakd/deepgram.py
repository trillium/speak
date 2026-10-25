"""Deepgram text-to-speech backend (alternate engine next to Kokoro).

Ported from the Deepgram speak call in the reviewed voice-mod plugin: a single
HTTPS POST to api.deepgram.com with ``Authorization: Token <key>``, body
``{"text": "..."}`` and the voice/format in query params. No other egress.

Design notes
------------
* Stdlib only at import time (numpy is imported lazily by the async adapter) so
  the CLI helper and the unit tests stay cheap.
* The API key is read from ``DEEPGRAM_API_KEY`` first, then from
  ``~/.config/deepgram-voice/env``. It is only ever placed in the request's
  ``Authorization`` header: never logged, never in an exception message, never
  persisted (state.json, history, event log, daemon stderr).
* The model/voice name and API version are validated against allow-patterns
  before they are interpolated into the URL.
* Redirects are refused: urllib would otherwise replay the Authorization header
  to whatever host a redirect pointed at.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

# --- Endpoint -----------------------------------------------------------------
# The host is deliberately NOT configurable: the key must only ever go here.
API_HOST = "api.deepgram.com"
DEFAULT_API_VERSION = "v1"
API_VERSION_ENV = "SPEAK_DEEPGRAM_API_VERSION"
API_VERSION_SETTING = "deepgram_api_version"

# --- Voices -------------------------------------------------------------------
DEFAULT_VOICE = "aura-2-thalia-en"
# Mirrors the reviewed plugin's allow-pattern: aura-/aura-2-/flux- style names.
_MODEL_RE = re.compile(r"^(aura|flux)(-[a-z0-9]+)+$")
_MODEL_MAX_LEN = 64
_API_VERSION_RE = re.compile(r"^v[0-9]{1,2}$")

# Common voices, for `speak --engine deepgram --voices`. Not exhaustive; any
# name matching the allow-pattern is passed through to the API.
KNOWN_VOICES = (
    "aura-2-thalia-en",
    "aura-2-andromeda-en",
    "aura-2-helena-en",
    "aura-2-luna-en",
    "aura-2-apollo-en",
    "aura-2-arcas-en",
    "aura-2-aries-en",
    "aura-2-orion-en",
    "aura-2-zeus-en",
    "aura-asteria-en",
    "aura-luna-en",
    "aura-orion-en",
)

# --- Audio format -------------------------------------------------------------
# Raw little-endian 16-bit mono PCM at the daemon's native 24 kHz, so the
# response drops straight into the existing playback/save path with no resample.
SAMPLE_RATE = 24000
ENCODING = "linear16"
CONTAINER = "none"

# Deepgram caps a single REST request at 2000 characters; stay under it.
MAX_CHARS_PER_REQUEST = 1900
# Speaking-rate multiplier range accepted by the API. Set DEEPGRAM_SPEED=off to
# stop sending the parameter at all (e.g. if a model rejects it).
SPEED_MIN, SPEED_MAX = 0.7, 1.5
SPEED_OFF_ENV = "DEEPGRAM_SPEED"
REQUEST_TIMEOUT_SECS = 30

# --- API key ------------------------------------------------------------------
KEY_ENV = "DEEPGRAM_API_KEY"
KEY_FILE = os.path.join("~", ".config", "deepgram-voice", "env")
# Printable, no whitespace/quotes: blocks header injection via a crafted file.
_KEY_RE = re.compile(r"^[A-Za-z0-9._~+/=-]{8,512}$")
_KEY_LINE_RE = re.compile(
    r"^\s*(?:export\s+)?" + KEY_ENV + r"\s*=\s*(?P<v>.*?)\s*$"
)


class DeepgramError(Exception):
    """Any Deepgram failure. Messages never contain the API key."""


# --- Validation ---------------------------------------------------------------

def is_valid_model(name) -> bool:
    return (
        isinstance(name, str)
        and len(name) <= _MODEL_MAX_LEN
        and _MODEL_RE.fullmatch(name) is not None
    )


def validate_model(name) -> str:
    if not is_valid_model(name):
        raise DeepgramError(
            "invalid Deepgram voice/model name (expected e.g. aura-2-thalia-en)"
        )
    return name


def resolve_api_version(env=None, settings=None) -> str:
    """API version: env > settings.json > v1. Validated before URL use."""
    env = os.environ if env is None else env
    version = env.get(API_VERSION_ENV) or (settings or {}).get(API_VERSION_SETTING)
    version = version or DEFAULT_API_VERSION
    if not isinstance(version, str) or not _API_VERSION_RE.fullmatch(version):
        raise DeepgramError("invalid Deepgram API version (expected e.g. v1)")
    return version


# --- API key ------------------------------------------------------------------

def _parse_key_file(path: str):
    """Extract DEEPGRAM_API_KEY from a KEY=VALUE env file (never `source`d)."""
    try:
        with open(os.path.expanduser(path), "r", encoding="utf-8") as f:
            for line in f:
                m = _KEY_LINE_RE.match(line)
                if m:
                    value = m.group("v")
                    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                        value = value[1:-1]
                    return value
    except OSError:
        return None
    return None


def load_api_key(env=None, key_file: str | None = None):
    """Return the API key (env var first, then the config file) or None.

    A key that fails the character allow-pattern is treated as absent rather
    than echoed anywhere.
    """
    env = os.environ if env is None else env
    for candidate in (env.get(KEY_ENV), _parse_key_file(key_file or KEY_FILE)):
        if candidate and _KEY_RE.fullmatch(candidate.strip()):
            return candidate.strip()
    return None


def key_source(env=None, key_file: str | None = None):
    """'env', 'file' or None. For diagnostics only; never returns the key."""
    env = os.environ if env is None else env
    val = env.get(KEY_ENV)
    if val and _KEY_RE.fullmatch(val.strip()):
        return "env"
    val = _parse_key_file(key_file or KEY_FILE)
    if val and _KEY_RE.fullmatch(val.strip()):
        return "file"
    return None


MISSING_KEY_MESSAGE = (
    f"Deepgram API key not found: set {KEY_ENV} or add "
    f"'{KEY_ENV}=<key>' to {KEY_FILE}"
)


# --- Request building ---------------------------------------------------------

def clamp_speed(speed) -> float | None:
    """Clamp to the API's accepted range; None when speed can't be used."""
    try:
        value = float(speed)
    except (TypeError, ValueError):
        return None
    if value != value or value <= 0:  # NaN or non-positive
        return None
    return max(SPEED_MIN, min(SPEED_MAX, value))


def build_url(model: str, api_version: str = DEFAULT_API_VERSION, speed=None) -> str:
    validate_model(model)
    if not _API_VERSION_RE.fullmatch(api_version or ""):
        raise DeepgramError("invalid Deepgram API version (expected e.g. v1)")
    params = [
        ("model", model),
        ("encoding", ENCODING),
        ("sample_rate", str(SAMPLE_RATE)),
        ("container", CONTAINER),
    ]
    clamped = clamp_speed(speed)
    if clamped is not None and os.environ.get(SPEED_OFF_ENV, "").lower() not in (
        "off", "0", "false", "no",
    ):
        params.append(("speed", f"{clamped:.2f}"))
    return f"https://{API_HOST}/{api_version}/speak?{urllib.parse.urlencode(params)}"


def chunk_text(text: str, limit: int = MAX_CHARS_PER_REQUEST) -> list[str]:
    """Split text into pieces <= limit chars, preferring sentence/word breaks."""
    text = text.strip()
    if len(text) <= limit:
        return [text] if text else []
    pieces: list[str] = []
    while len(text) > limit:
        window = text[:limit]
        cut = max(window.rfind(". "), window.rfind("! "), window.rfind("? "))
        if cut < limit // 2:
            cut = window.rfind(" ")
        cut = cut + 1 if cut > 0 else limit
        pieces.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        pieces.append(text)
    return [p for p in pieces if p]


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse redirects so the Authorization header can't follow one off-host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _default_transport(request, timeout):
    opener = urllib.request.build_opener(_NoRedirect)
    return opener.open(request, timeout=timeout)


def _error_detail(exc: urllib.error.HTTPError, key: str) -> str:
    """Short, key-free description of an HTTP error body."""
    try:
        body = exc.read(512).decode("utf-8", "replace")
    except Exception:
        body = ""
    body = " ".join(body.split())[:200].replace(key, "[redacted]")
    return f"HTTP {exc.code}" + (f": {body}" if body else "")


def synthesize_pcm(
    text: str,
    model: str = DEFAULT_VOICE,
    speed=None,
    *,
    api_key: str | None = None,
    api_version: str | None = None,
    transport=None,
    timeout: float = REQUEST_TIMEOUT_SECS,
) -> bytes:
    """Synthesize text via Deepgram; returns raw s16le mono 24 kHz PCM bytes.

    `transport(request, timeout)` returns a file-like response; it exists so
    tests can stub the network. Raises DeepgramError on any failure.
    """
    key = api_key or load_api_key()
    if not key:
        raise DeepgramError(MISSING_KEY_MESSAGE)
    from . import config  # lazy: config needs $USER, irrelevant to pure helpers

    version = api_version or resolve_api_version(settings=config.load_settings())
    url = build_url(model, version, speed)
    pieces = chunk_text(text)
    if not pieces:
        raise DeepgramError("no text to synthesize")

    send = transport or _default_transport
    out = bytearray()
    for piece in pieces:
        request = urllib.request.Request(
            url,
            data=json.dumps({"text": piece}).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Token {key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with send(request, timeout) as resp:
                data = resp.read()
        except urllib.error.HTTPError as e:
            raise DeepgramError(f"Deepgram request failed ({_error_detail(e, key)})") from None
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            # str(e) can embed the URL but never headers; reason only.
            reason = getattr(e, "reason", e)
            raise DeepgramError(f"Deepgram request failed ({type(reason).__name__})") from None
        if not data:
            raise DeepgramError("Deepgram returned no audio")
        out += data
    if len(out) % 2:  # s16le: drop a stray odd byte rather than corrupt a frame
        del out[-1]
    return bytes(out)


def validate_enqueue_request(request: dict, env=None, key_file: str | None = None):
    """Daemon-side gate for an enqueue request. Returns an error string or None.

    Kokoro (the default, engine absent) is never rejected here, so existing
    callers are unaffected.
    """
    engine = request.get("engine") or "kokoro"
    if engine == "kokoro":
        return None
    if engine != "deepgram":
        return f"unknown engine {str(engine)[:32]!r} (daemon supports kokoro, deepgram)"
    if not is_valid_model(request.get("voice")):
        return "invalid Deepgram voice/model name (expected e.g. aura-2-thalia-en)"
    if load_api_key(env=env, key_file=key_file) is None:
        return MISSING_KEY_MESSAGE
    return None


# --- Daemon adapter -----------------------------------------------------------

class DeepgramTTS:
    """Drop-in for Kokoro's ``create_stream`` as used by the renderer."""

    def __init__(self, transport=None):
        self._transport = transport

    async def create_stream(self, text, voice, speed, lang="en-us", trim=False):
        """Yield (float32 mono audio @ 24 kHz, sample_rate), like Kokoro.

        One buffered REST call per clause (the renderer already splits text
        into clauses); the blocking HTTP call runs off the event loop.
        """
        import numpy as np

        pcm = await asyncio.to_thread(
            synthesize_pcm, text, voice, speed, transport=self._transport
        )
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        yield audio, SAMPLE_RATE


# --- CLI helper (bin/speak-deepgram) -----------------------------------------

def _write_wav(path: str, pcm: bytes) -> None:
    import wave

    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)


def main(argv=None, stdin=None, stdout=None) -> int:
    """Read text on stdin; write WAV to --save FILE or raw PCM to stdout."""
    import argparse

    parser = argparse.ArgumentParser(prog="speak-deepgram")
    parser.add_argument("--voice", default=DEFAULT_VOICE)
    parser.add_argument("--speed", type=float, default=None)
    parser.add_argument("--save", default=None, help="write a WAV file instead of raw PCM to stdout")
    args = parser.parse_args(argv)

    text = (stdin or sys.stdin).read().strip()
    if not text:
        print("speak-deepgram: empty text", file=sys.stderr)
        return 1
    try:
        pcm = synthesize_pcm(text, args.voice, args.speed)
        if args.save:
            _write_wav(args.save, pcm)
            if os.path.getsize(args.save) == 0:
                raise DeepgramError(f"wrote nothing to {args.save}")
        else:
            out = stdout or sys.stdout.buffer
            out.write(pcm)
            out.flush()
    except DeepgramError as e:
        print(f"speak-deepgram: {e}", file=sys.stderr)
        return 1
    except OSError as e:
        print(f"speak-deepgram: output error: {e.strerror or type(e).__name__}", file=sys.stderr)
        return 1
    return 0
