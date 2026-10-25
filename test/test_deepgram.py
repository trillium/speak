"""Tests for the Deepgram TTS engine wiring. No network: the transport is stubbed.

Run:  python3 -m unittest discover -s test -v      (or: pytest test)
"""

import asyncio
import io
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
import urllib.error

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lib"))
os.environ.setdefault("USER", "speaktest")  # speakd.config reads $USER at import

from speakd import deepgram as dg  # noqa: E402

FAKE_KEY = "test-key-0123456789abcdef"  # obviously fake; never a real credential


class FakeResponse:
    def __init__(self, data=b"", status=200):
        self._data = data
        self.status = status

    def read(self, *_):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class RecordingTransport:
    """Stub transport: records requests, replays canned PCM or raises."""

    def __init__(self, pcm=b"\x01\x00\x02\x00", error=None):
        self.requests = []
        self.pcm = pcm
        self.error = error

    def __call__(self, request, timeout):
        self.requests.append(request)
        if self.error:
            raise self.error
        return FakeResponse(self.pcm)


def synth(text="hello", transport=None, **kw):
    kw.setdefault("api_key", FAKE_KEY)
    kw.setdefault("api_version", "v1")
    return dg.synthesize_pcm(text, transport=transport or RecordingTransport(), **kw)


class VoiceValidation(unittest.TestCase):
    def test_accepts_aura_and_flux_names(self):
        for name in ("aura-2-thalia-en", "aura-asteria-en", "flux-general-en"):
            self.assertTrue(dg.is_valid_model(name), name)

    def test_rejects_injection_and_junk(self):
        bad = [
            "", None, 42, "af_heart", "Aura-2-thalia-en", "aura", "aura-",
            "aura-2-thalia-en&speed=9", "aura-2-thalia-en?x=1", "aura-2/../x",
            "aura-2-thalia-en\nHost: evil", "aura-2 thalia", "aura-" + "a" * 80,
            "https://evil.example/aura-2-x",
        ]
        for name in bad:
            self.assertFalse(dg.is_valid_model(name), repr(name))
            with self.assertRaises(dg.DeepgramError):
                dg.validate_model(name)

    def test_api_version_validation(self):
        self.assertEqual(dg.resolve_api_version(env={}), "v1")
        self.assertEqual(dg.resolve_api_version(env={dg.API_VERSION_ENV: "v2"}), "v2")
        self.assertEqual(
            dg.resolve_api_version(env={}, settings={dg.API_VERSION_SETTING: "v2"}), "v2"
        )
        for bad in ("v1/../x", "1", "v", "evil.com", "v100"):
            with self.assertRaises(dg.DeepgramError):
                dg.resolve_api_version(env={dg.API_VERSION_ENV: bad})


class UrlBuilding(unittest.TestCase):
    def test_url_shape(self):
        url = dg.build_url("aura-2-thalia-en", "v1", 1.26)
        self.assertTrue(url.startswith("https://api.deepgram.com/v1/speak?"))
        for part in ("model=aura-2-thalia-en", "encoding=linear16",
                     "sample_rate=24000", "container=none", "speed=1.26"):
            self.assertIn(part, url)

    def test_v2_is_configurable(self):
        self.assertIn("/v2/speak?", dg.build_url("aura-2-thalia-en", "v2"))

    def test_speed_is_clamped_and_optional(self):
        self.assertIn("speed=1.50", dg.build_url("aura-2-thalia-en", "v1", 9))
        self.assertIn("speed=0.70", dg.build_url("aura-2-thalia-en", "v1", 0.1))
        self.assertNotIn("speed=", dg.build_url("aura-2-thalia-en", "v1", None))
        self.assertNotIn("speed=", dg.build_url("aura-2-thalia-en", "v1", "nope"))

    def test_speed_can_be_disabled(self):
        old = os.environ.get(dg.SPEED_OFF_ENV)
        os.environ[dg.SPEED_OFF_ENV] = "off"
        try:
            self.assertNotIn("speed=", dg.build_url("aura-2-thalia-en", "v1", 1.2))
        finally:
            if old is None:
                os.environ.pop(dg.SPEED_OFF_ENV)
            else:
                os.environ[dg.SPEED_OFF_ENV] = old

    def test_bad_model_never_reaches_url(self):
        with self.assertRaises(dg.DeepgramError):
            dg.build_url("aura-x&evil=1", "v1")


class ApiKey(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.file = os.path.join(self.tmp.name, "env")

    def write(self, text):
        with open(self.file, "w") as f:
            f.write(text)

    def test_env_wins_over_file(self):
        self.write(f"DEEPGRAM_API_KEY=file-key-0123456789\n")
        self.assertEqual(dg.load_api_key({dg.KEY_ENV: FAKE_KEY}, self.file), FAKE_KEY)
        self.assertEqual(dg.key_source({dg.KEY_ENV: FAKE_KEY}, self.file), "env")

    def test_file_fallback_formats(self):
        for line in ("DEEPGRAM_API_KEY=file-key-0123456789",
                     "export DEEPGRAM_API_KEY=file-key-0123456789",
                     'DEEPGRAM_API_KEY="file-key-0123456789"',
                     "# c\nDEEPGRAM_API_KEY='file-key-0123456789'  "):
            self.write(line + "\n")
            self.assertEqual(dg.load_api_key({}, self.file), "file-key-0123456789", line)
            self.assertEqual(dg.key_source({}, self.file), "file")

    def test_missing_everywhere(self):
        self.assertIsNone(dg.load_api_key({}, self.file))
        self.assertIsNone(dg.key_source({}, self.file))

    def test_rejects_keys_with_unsafe_characters(self):
        self.write("DEEPGRAM_API_KEY=abc def 123456789\n")
        self.assertIsNone(dg.load_api_key({}, self.file))
        self.assertIsNone(dg.load_api_key({dg.KEY_ENV: "key\r\nX-Evil: 1 12345"}, self.file))

    def test_missing_key_message_has_no_secret_and_names_sources(self):
        self.assertIn("DEEPGRAM_API_KEY", dg.MISSING_KEY_MESSAGE)
        self.assertIn(".config/deepgram-voice/env", dg.MISSING_KEY_MESSAGE)


class Synthesis(unittest.TestCase):
    def test_request_shape(self):
        t = RecordingTransport(pcm=b"\x00\x01" * 4)
        pcm = synth("Hello there", t, model="aura-2-thalia-en", speed=1.2)
        self.assertEqual(pcm, b"\x00\x01" * 4)
        self.assertEqual(len(t.requests), 1)
        req = t.requests[0]
        self.assertEqual(req.get_method(), "POST")
        self.assertTrue(req.full_url.startswith("https://api.deepgram.com/v1/speak?"))
        self.assertEqual(req.get_header("Authorization"), f"Token {FAKE_KEY}")
        self.assertEqual(req.get_header("Content-type"), "application/json")
        self.assertEqual(json.loads(req.data), {"text": "Hello there"})
        # The key travels only in the header, never in the URL or body.
        self.assertNotIn(FAKE_KEY, req.full_url)
        self.assertNotIn(FAKE_KEY.encode(), req.data)

    def test_v2_endpoint(self):
        t = RecordingTransport()
        synth("hi", t, api_version="v2")
        self.assertIn("/v2/speak?", t.requests[0].full_url)

    def test_invalid_voice_makes_no_request(self):
        t = RecordingTransport()
        with self.assertRaises(dg.DeepgramError):
            synth("hi", t, model="aura-2-x&y=1")
        self.assertEqual(t.requests, [])

    def test_missing_key_makes_no_request(self):
        t = RecordingTransport()
        old = os.environ.pop(dg.KEY_ENV, None)
        orig = dg._parse_key_file
        dg._parse_key_file = lambda path: None
        try:
            with self.assertRaises(dg.DeepgramError) as cm:
                dg.synthesize_pcm("hi", transport=t, api_key=None, api_version="v1")
        finally:
            dg._parse_key_file = orig
            if old is not None:
                os.environ[dg.KEY_ENV] = old
        self.assertIn("API key not found", str(cm.exception))
        self.assertEqual(t.requests, [])

    def test_long_text_is_chunked(self):
        t = RecordingTransport(pcm=b"\x01\x00")
        text = ("This is a sentence. " * 300).strip()
        pcm = synth(text, t)
        self.assertGreater(len(t.requests), 1)
        self.assertEqual(len(pcm), 2 * len(t.requests))
        for r in t.requests:
            self.assertLessEqual(len(json.loads(r.data)["text"]), dg.MAX_CHARS_PER_REQUEST)

    def test_chunk_text_without_spaces_hard_splits(self):
        pieces = dg.chunk_text("x" * 4000, 1900)
        self.assertEqual("".join(pieces), "x" * 4000)
        self.assertTrue(all(len(p) <= 1900 for p in pieces))

    def test_http_error_is_reported_without_key(self):
        body = io.BytesIO(json.dumps({"err_msg": f"bad token {FAKE_KEY}"}).encode())
        err = urllib.error.HTTPError("https://api.deepgram.com/v1/speak", 401, "x", {}, body)
        with self.assertRaises(dg.DeepgramError) as cm:
            synth("hi", RecordingTransport(error=err))
        self.assertIn("HTTP 401", str(cm.exception))
        self.assertNotIn(FAKE_KEY, str(cm.exception))

    def test_network_error_is_reported_without_key_or_url(self):
        err = urllib.error.URLError(ConnectionRefusedError("nope"))
        with self.assertRaises(dg.DeepgramError) as cm:
            synth("hi", RecordingTransport(error=err))
        self.assertIn("ConnectionRefusedError", str(cm.exception))
        self.assertNotIn(FAKE_KEY, str(cm.exception))

    def test_empty_audio_is_an_error(self):
        with self.assertRaises(dg.DeepgramError):
            synth("hi", RecordingTransport(pcm=b""))

    def test_odd_trailing_byte_is_dropped(self):
        self.assertEqual(len(synth("hi", RecordingTransport(pcm=b"\x01\x00\x02"))), 2)

    def test_redirects_are_refused(self):
        handler = dg._NoRedirect()
        self.assertIsNone(
            handler.redirect_request(None, None, 302, "Found", {}, "https://evil.example/")
        )


class EnqueueValidation(unittest.TestCase):
    def check(self, request, env=None):
        with tempfile.TemporaryDirectory() as d:
            return dg.validate_enqueue_request(request, env=env or {}, key_file=os.path.join(d, "none"))

    def test_kokoro_and_absent_engine_pass_untouched(self):
        self.assertIsNone(self.check({"voice": "af_heart"}))
        self.assertIsNone(self.check({"voice": "af_heart", "engine": "kokoro"}))

    def test_deepgram_ok_with_key(self):
        self.assertIsNone(
            self.check({"engine": "deepgram", "voice": "aura-2-thalia-en"}, {dg.KEY_ENV: FAKE_KEY})
        )

    def test_deepgram_without_key_is_rejected_loudly(self):
        self.assertIn("API key", self.check({"engine": "deepgram", "voice": "aura-2-thalia-en"}))

    def test_deepgram_bad_voice_is_rejected(self):
        err = self.check({"engine": "deepgram", "voice": "af_heart"}, {dg.KEY_ENV: FAKE_KEY})
        self.assertIn("invalid Deepgram voice", err)

    def test_unknown_engine_is_rejected(self):
        self.assertIn("unknown engine", self.check({"engine": "piper", "voice": "x"}))


def pcm_for(samples):
    import struct

    return struct.pack(f"<{len(samples)}h", *samples)


class DaemonAdapter(unittest.TestCase):
    def test_create_stream_yields_float32_at_24k(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy not installed")
        t = RecordingTransport(pcm=pcm_for([0, 16384, -16384, 32767]))
        os.environ[dg.KEY_ENV] = FAKE_KEY
        os.environ[dg.API_VERSION_ENV] = "v1"
        self.addCleanup(os.environ.pop, dg.KEY_ENV, None)
        self.addCleanup(os.environ.pop, dg.API_VERSION_ENV, None)

        async def run():
            return [c async for c in dg.DeepgramTTS(transport=t).create_stream(
                "hi", "aura-2-thalia-en", 1.26, "en-us", trim=False)]

        chunks = asyncio.run(run())
        self.assertEqual(len(chunks), 1)
        audio, sr = chunks[0]
        self.assertEqual(sr, 24000)
        self.assertEqual(audio.dtype, np.float32)
        self.assertAlmostEqual(float(audio[1]), 0.5, places=3)
        self.assertEqual(len(t.requests), 1)


class EngineSelection(unittest.TestCase):
    def test_backend_for_routes_by_engine(self):
        try:
            from speakd.synthesis import SynthesisEngine
        except ImportError:
            self.skipTest("kokoro_onnx not installed")
        sentinel = object()
        engine = SynthesisEngine(sentinel, cache=None)
        self.assertIs(engine.backend_for(None), sentinel)
        self.assertIs(engine.backend_for("kokoro"), sentinel)
        self.assertIs(engine.backend_for("deepgram"), engine.deepgram)

    def test_prefetch_uses_selected_backend(self):
        try:
            from speakd.renderer import prefetch_first_chunk
        except ImportError:
            self.skipTest("renderer dependencies not installed")

        calls = []

        class Backend:
            def __init__(self, name):
                self.name = name

            async def create_stream(self, text, voice, speed, lang, trim=False):
                calls.append((self.name, text, voice))
                yield [0.0], 24000

        class Synth:
            kokoro, deepgram = Backend("kokoro"), Backend("deepgram")

            def backend_for(self, engine):
                return self.deepgram if engine == "deepgram" else self.kokoro

        asyncio.run(prefetch_first_chunk(Synth(), "Hi there.", "aura-2-thalia-en", 1.0, "en-us", engine="deepgram"))
        asyncio.run(prefetch_first_chunk(Synth(), "Hi there.", "af_heart", 1.0, "en-us"))
        self.assertEqual([c[0] for c in calls], ["deepgram", "kokoro"])


class RenderPath(unittest.TestCase):
    def test_render_speech_plays_deepgram_audio_through_existing_path(self):
        try:
            import numpy as np
            from speakd import renderer
            from speakd.synthesis import SynthesisEngine
        except ImportError:
            self.skipTest("daemon dependencies not installed")

        class Boom:  # Kokoro must not be touched for a deepgram request
            def create_stream(self, *a, **k):
                raise AssertionError("kokoro used for a deepgram request")

        class FakeAudio:
            is_alive = True

            def __init__(self):
                self.writes = []

            async def write_pcm(self, pcm, skip_flag_fn=None):
                self.writes.append(pcm)

        transport = RecordingTransport(pcm=pcm_for([8000] * 2400))
        synth_engine = SynthesisEngine(Boom(), cache=None)
        synth_engine.deepgram = dg.DeepgramTTS(transport=transport)
        audio = FakeAudio()
        os.environ[dg.KEY_ENV] = FAKE_KEY
        os.environ[dg.API_VERSION_ENV] = "v1"
        self.addCleanup(os.environ.pop, dg.KEY_ENV, None)
        self.addCleanup(os.environ.pop, dg.API_VERSION_ENV, None)
        orig_log = renderer.log_event
        renderer.log_event = lambda *a, **k: None  # keep the real event log clean
        self.addCleanup(setattr, renderer, "log_event", orig_log)

        done = asyncio.run(renderer.render_speech(
            {"text": "Hello there. How are you?", "voice": "aura-2-thalia-en",
             "speed": 1.26, "engine": "deepgram", "_queue_id": 1},
            None, synth_engine, audio, lambda: False, lambda f: None,
        ))
        self.assertEqual(done, 2)  # two clauses -> two Deepgram requests -> two writes
        self.assertEqual(len(transport.requests), 2)
        self.assertEqual(len(audio.writes), 2)
        self.assertTrue(all(len(w) > 0 for w in audio.writes))
        for r in transport.requests:
            self.assertIn("model=aura-2-thalia-en", r.full_url)


class CliWiring(unittest.TestCase):
    """bin/speak flag/settings plumbing. Uses --voices so no daemon, model or network."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = self.tmp.name
        self.settings = os.path.join(self.home, "settings")
        os.makedirs(self.settings)

    def env(self, **extra):
        env = {k: v for k, v in os.environ.items() if k not in (dg.KEY_ENV, "SPEAK_ENGINE")}
        env.update(HOME=self.home, SPEAK_SETTINGS_DIR=self.settings, USER=env.get("USER", "speaktest"))
        env.update(extra)
        return env

    def speak(self, *args, **extra):
        if not shutil_which("uv"):
            self.skipTest("uv not installed (bin/speak requires it)")
        return subprocess.run(
            ["bash", str(ROOT / "bin" / "speak"), *args],
            capture_output=True, text=True, env=self.env(**extra), timeout=60,
            stdin=subprocess.DEVNULL,
        )

    def write_settings(self, **kw):
        with open(os.path.join(self.settings, "settings.json"), "w") as f:
            json.dump(kw, f, indent=2)

    def test_engine_flag_accepts_deepgram(self):
        r = self.speak("--engine", "deepgram", "--voices")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("aura-2-thalia-en", r.stdout)

    def test_unknown_engine_names_all_three(self):
        r = self.speak("--engine", "bogus", "--voices", "x")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("kokoro, piper or deepgram", r.stderr)

    def test_settings_json_selects_engine(self):
        self.write_settings(engine="deepgram")
        r = self.speak("--voices")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("aura-2-thalia-en", r.stdout)

    def test_settings_json_compact_form_is_read(self):
        with open(os.path.join(self.settings, "settings.json"), "w") as f:
            f.write('{"engine":"deepgram","output_device":"Jabra"}')
        r = self.speak("--voices")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("aura-2-thalia-en", r.stdout)

    def test_env_beats_settings_and_flag_beats_env(self):
        self.write_settings(engine="piper")
        r = self.speak("--voices", SPEAK_ENGINE="deepgram")
        self.assertIn("aura-2-thalia-en", r.stdout)
        self.write_settings(engine="deepgram")
        r = self.speak("--engine", "deepgram", "--voices", SPEAK_ENGINE="piper")
        self.assertIn("aura-2-thalia-en", r.stdout)

    def test_unknown_settings_engine_is_ignored_not_fatal(self):
        self.write_settings(engine="bogus")
        r = self.speak("--engine", "deepgram", "--voices")  # flag wins; setting never consulted
        self.assertEqual(r.returncode, 0, r.stderr)
        # No flag: the bogus setting warns and kokoro stays the engine. --save
        # fails fast on the (absent) kokoro model files, so nothing heavy runs.
        r = self.speak("--save", os.path.join(self.home, "o.wav"), "hello")
        self.assertIn("ignoring unknown configured engine 'bogus'", r.stderr)
        self.assertIn("kokoro model/voices missing", r.stderr)

    def test_save_without_key_fails_loudly_and_writes_nothing(self):
        out = os.path.join(self.home, "out.wav")
        r = self.speak("--engine", "deepgram", "--save", out, "hello")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("API key not found", r.stderr)
        self.assertFalse(os.path.exists(out))

    def test_save_bad_voice_is_rejected_before_any_request(self):
        out = os.path.join(self.home, "out.wav")
        r = self.speak("--engine", "deepgram", "--voice", "af_heart", "--save", out, "hello",
                       DEEPGRAM_API_KEY=FAKE_KEY)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("invalid Deepgram voice", r.stderr)
        self.assertNotIn(FAKE_KEY, r.stderr + r.stdout)
        self.assertFalse(os.path.exists(out))


class SaveWav(unittest.TestCase):
    def test_main_save_writes_real_wav(self):
        import wave

        pcm = pcm_for([0, 1000, -1000, 2000] * 100)
        orig = dg.synthesize_pcm
        dg.synthesize_pcm = lambda text, voice, speed: pcm
        self.addCleanup(setattr, dg, "synthesize_pcm", orig)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "o.wav")
            rc = dg.main(["--voice", "aura-2-thalia-en", "--save", path], stdin=io.StringIO("hi"))
            self.assertEqual(rc, 0)
            with wave.open(path, "rb") as w:
                self.assertEqual((w.getnchannels(), w.getsampwidth(), w.getframerate()), (1, 2, 24000))
                self.assertEqual(w.readframes(w.getnframes()), pcm)

    def test_main_raw_writes_pcm_to_stdout(self):
        pcm = pcm_for([1, 2, 3])
        orig = dg.synthesize_pcm
        dg.synthesize_pcm = lambda text, voice, speed: pcm
        self.addCleanup(setattr, dg, "synthesize_pcm", orig)
        out = io.BytesIO()
        self.assertEqual(dg.main([], stdin=io.StringIO("hi"), stdout=out), 0)
        self.assertEqual(out.getvalue(), pcm)

    def test_main_reports_errors_without_traceback(self):
        def boom(text, voice, speed):
            raise dg.DeepgramError("Deepgram request failed (HTTP 401)")

        orig = dg.synthesize_pcm
        dg.synthesize_pcm = boom
        self.addCleanup(setattr, dg, "synthesize_pcm", orig)
        self.assertEqual(dg.main([], stdin=io.StringIO("hi"), stdout=io.BytesIO()), 1)


def shutil_which(cmd):
    import shutil

    return shutil.which(cmd)


if __name__ == "__main__":
    unittest.main()
