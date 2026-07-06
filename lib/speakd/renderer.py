"""Speech rendering with clause-level streaming and configurable trim.

Splits text into clauses before synthesis so the first clause can be
prefetched during caller tone playback. Each clause's audio is trimmed
of Kokoro's built-in silence padding (~280ms lead, ~360ms trail) and
replaced with punctuation-appropriate gaps from config/trim.yaml.

Config is re-read on every synthesis so edits take effect immediately.
"""

import asyncio
import os
import pathlib
import re
import sys
import time
from typing import AsyncIterator

import numpy as np
import yaml

from .config import SAMPLE_RATE
from .playback_device import AudioOutputStream
from .protocol import log_event
from .text import split_clauses

# Per-chunk and prefetch timing prints are off by default; set
# SPEAK_DEBUG_TIMING=1 to emit them. The single per-request START/DONE lines
# are always logged.
DEBUG_TIMING = os.environ.get("SPEAK_DEBUG_TIMING") == "1"

# Silence detection threshold: fraction of peak amplitude
_SILENCE_THRESH = 0.001

_TRIM_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "config", "trim.yaml"
)


# (mtime, (gaps, default)) of the last successful trim.yaml parse, or None.
_trim_config_cache: tuple[float, tuple[dict, int]] | None = None


def _load_trim_config():
    """Load trim.yaml, returning (gaps_dict, default_gap_ms).

    Parsed once and cached, keyed on the file's mtime, so it is not re-read
    per clause. Editing trim.yaml bumps the mtime and takes effect on the next
    synthesis — hot-reload semantics are preserved.
    """
    global _trim_config_cache
    try:
        mtime = os.stat(_TRIM_CONFIG_PATH).st_mtime
    except FileNotFoundError:
        _trim_config_cache = None
        return {}, 200

    if _trim_config_cache is not None and _trim_config_cache[0] == mtime:
        return _trim_config_cache[1]

    try:
        with open(_TRIM_CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        gaps = cfg.get("gaps", {})
        default = cfg.get("default_gap", 200)
    except (FileNotFoundError, yaml.YAMLError):
        return {}, 200

    _trim_config_cache = (mtime, (gaps, default))
    return gaps, default


def _find_voice_bounds(audio):
    """Find first and last sample above silence threshold. Returns (start, end) indices."""
    abs_audio = np.abs(audio)
    peak = np.max(abs_audio)
    if peak == 0:
        return 0, len(audio)
    threshold = peak * _SILENCE_THRESH
    above = np.where(abs_audio > threshold)[0]
    if len(above) == 0:
        return 0, len(audio)
    return int(above[0]), int(above[-1]) + 1


def audio_to_pcm(audio, gain: float = 1.0) -> bytes:
    """Convert float32 [-1.0, 1.0] audio to 16-bit little-endian PCM bytes.

    Gain is applied to the float samples BEFORE int16 quantization so there is
    a single rounding step. The old path quantized to int16, cast back to
    float32, multiplied by gain, then re-quantized — double-rounding that this
    replaces. Samples are clipped to the int16 range.
    """
    samples = audio
    if gain != 1.0:
        samples = samples * gain
    pcm = np.clip(samples * 32767.0, -32768.0, 32767.0).astype(np.int16)
    return pcm.tobytes()


def trim_clause_audio(audio, split_char, prev_split_char, is_first):
    """Strip silence from audio and add punctuation-appropriate padding.

    Returns trimmed audio as float32 array.
    """
    gaps, default_gap = _load_trim_config()

    start, end = _find_voice_bounds(audio)
    voice = audio[start:end]

    # Trail padding based on THIS clause's ending punctuation
    trail_ms = gaps.get(split_char, default_gap) / 2
    # Lead padding based on PREVIOUS clause's ending punctuation
    if is_first:
        lead_ms = 10  # minimal lead on first clause
    else:
        lead_ms = gaps.get(prev_split_char, default_gap) / 2

    lead_samples = int(SAMPLE_RATE * lead_ms / 1000)
    trail_samples = int(SAMPLE_RATE * trail_ms / 1000)

    return np.concatenate([
        np.zeros(lead_samples, dtype=np.float32),
        voice,
        np.zeros(trail_samples, dtype=np.float32),
    ])


_SYS_CLIPS_DIR = pathlib.Path.home() / ".local/share/speak/sys-clips"
_SYS_SENTINEL = re.compile(r'(<<sys:[^>]+>>)')


def _split_sys_segments(text: str) -> list[tuple[str, str]]:
    """Split text at <<sys:slug>> sentinels.

    Returns list of ('text', content) or ('sys', slug) tuples.
    """
    parts = _SYS_SENTINEL.split(text)
    result = []
    for part in parts:
        if m := re.match(r'<<sys:([^>]+)>>', part):
            result.append(('sys', m.group(1)))
        elif part.strip():
            result.append(('text', part))
    return result


async def _play_sys_clip(
    slug: str,
    audio_stream: AudioOutputStream,
    skip_flag_fn,
    gain: float,
) -> bool:
    """Play a pre-built sys clip. Returns False if clip not found (caller should synthesize fallback)."""
    clip_path = _SYS_CLIPS_DIR / f"{slug}.pcm"
    if not clip_path.exists():
        return False
    pcm = clip_path.read_bytes()
    if gain != 1.0:
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        samples = np.clip(samples * gain, -32767, 32767).astype(np.int16)
        pcm = samples.tobytes()
    await audio_stream.write_pcm(pcm, skip_flag_fn=skip_flag_fn)
    return True


async def prefetch_first_chunk(synth, text, voice_name, speed, lang):
    """Synthesize the first clause of the first text segment concurrently with tone playback.

    Splits text at sys sentinels first. Prefetches only the first clause of the
    first text segment. If text starts with a sys sentinel, returns ([], []) since
    sys clips don't benefit from prefetch.

    Returns (first_audio_chunks, remaining_clauses) where first_audio_chunks is
    a list of (audio, sr) tuples from the first clause's stream, and
    remaining_clauses is a list of strings still to be synthesized.
    """
    t0 = time.monotonic()
    label = text[:40].replace('\n', ' ')

    # Find the first text segment to prefetch
    segments = _split_sys_segments(text)
    first_text = next((content for kind, content in segments if kind == 'text'), None)
    if not first_text:
        print(f"speak-daemon: prefetch_first_chunk SKIP — no text segment in \"{label}\"", file=sys.stderr)
        return [], []

    clauses = split_clauses(first_text)
    if not clauses:
        print(f"speak-daemon: prefetch_first_chunk EMPTY text=\"{label}\"", file=sys.stderr)
        return [], []

    first_clause = clauses[0]
    remaining = clauses[1:]

    if DEBUG_TIMING:
        print(
            f"speak-daemon: prefetch_first_chunk STARTED voice={voice_name} "
            f"clauses={len(clauses)} first=\"{first_clause[:40]}\"",
            file=sys.stderr,
        )

    first_chunks = []
    async for audio, sr in synth.kokoro.create_stream(
        first_clause, voice_name, speed, lang, trim=False
    ):
        first_chunks.append((audio, sr))

    if DEBUG_TIMING:
        elapsed_ms = (time.monotonic() - t0) * 1000
        print(
            f"speak-daemon: prefetch_first_chunk DONE {elapsed_ms:.0f}ms "
            f"chunks={len(first_chunks)} remaining_clauses={len(remaining)}",
            file=sys.stderr,
        )
    return first_chunks, remaining


async def render_speech(
    request: dict,
    loop: asyncio.AbstractEventLoop,
    synth,
    audio_stream: AudioOutputStream,
    skip_flag_fn,
    bg_task_tracker,
    prefetch=None,
    on_first_write=None,
    resume_from_clause: int = 0,
    on_clause_start=None,
) -> int:
    """Stream speech from Kokoro to the audio device, clause by clause.

    If prefetch is provided, it should be (first_chunks, remaining_clauses)
    from prefetch_first_chunk(). The first clause audio is played immediately,
    then remaining clauses are synthesized and played sequentially.

    Without prefetch, the full text is split into clauses and each is
    synthesized independently.

    If resume_from_clause > 0, clauses before that index are skipped (used
    when resuming from a pause mid-utterance).

    Returns the number of clauses completed (chunks_done).
    """
    text = request.get("text", "").strip()
    voice_name = request.get("voice", "af_heart")
    speed = request.get("speed", 1.0)
    lang = request.get("lang", "en-us")
    qid = request.get("_queue_id", "?")
    caller = request.get("caller", "")

    # Voice and gain already resolved by playback.py via VoicePool
    voice_name = request.get("_resolved_voice", voice_name)
    gain = request.get("_gain", 1.0)

    if not text:
        return 0

    item_t0 = time.monotonic()
    label = text[:60].replace('\n', ' ')
    caller_tag = f" caller={caller}" if caller else ""
    gain_tag = f" gain={gain}" if gain != 1.0 else ""
    print(f"speak-daemon: [q#{qid}] START  voice={voice_name} speed={speed}{caller_tag}{gain_tag} \"{label}\"", file=sys.stderr)
    log_event("request_start", qid=qid, voice=voice_name, speed=speed,
              caller=caller, text=label)

    total_audio_secs = 0.0
    chunks_done = 0
    chunk_idx = 0

    prev_split_char = None  # tracks previous clause's ending punctuation

    async def _play_audio(audio):
        """Convert float32 audio to PCM (gain applied pre-quantization) and write."""
        nonlocal total_audio_secs, chunks_done, chunk_idx

        pcm = audio_to_pcm(audio, gain)
        dur = (len(pcm) // 2) / SAMPLE_RATE

        log_event("chunk_ready", qid=qid, chunk=chunk_idx + 1,
                  audio_secs=round(dur, 2), audio_bytes=len(pcm))

        if chunk_idx == 0 and on_first_write is not None:
            on_first_write()

        await audio_stream.write_pcm(pcm, skip_flag_fn=skip_flag_fn)

        total_audio_secs += dur
        chunks_done += 1
        chunk_idx += 1

        if DEBUG_TIMING:
            print(
                f"speak-daemon: [q#{qid}]   chunk {chunk_idx} "
                f"audio={dur:.2f}s",
                file=sys.stderr,
            )

    def _get_split_char(clause):
        """Get the trailing punctuation from a clause."""
        if clause and clause[-1] in ".!?,;:-\u2014":
            return clause[-1]
        return ""

    async def _synthesize_and_play_clause(clause, is_first):
        """Synthesize a clause, trim silence, add punctuation gaps, play."""
        nonlocal prev_split_char

        split_char = _get_split_char(clause)

        # Collect all audio chunks from this clause into one array
        clause_audio = []
        async for audio, sr in synth.kokoro.create_stream(
            clause, voice_name, speed, lang, trim=False
        ):
            if skip_flag_fn():
                return False
            clause_audio.append(audio.squeeze())

        if not clause_audio:
            return True

        # Concatenate chunks (usually just one for text < 510 phonemes)
        full_audio = np.concatenate(clause_audio)

        # Trim silence and add punctuation-appropriate padding
        trimmed = trim_clause_audio(full_audio, split_char, prev_split_char, is_first)

        await _play_audio(trimmed)
        prev_split_char = split_char
        return True

    if resume_from_clause > 0:
        print(
            f"speak-daemon: [q#{qid}] RESUMING from clause {resume_from_clause}",
            file=sys.stderr,
        )

    # Split text into sys-sentinel and text segments
    segments = _split_sys_segments(text)
    clause_offset = 0  # global clause index across all text segments

    if prefetch is not None:
        first_chunks, remaining_clauses = prefetch
        # The prefetch covers the first clause of the first text segment.
        # We play the prefetched audio first, then iterate segments normally
        # (skipping the first clause of the first text segment since it's already done).
        prefetch_consumed = False

        for seg_kind, seg_content in segments:
            if skip_flag_fn():
                break

            if seg_kind == 'sys':
                # Play sys clip (or fall back to synthesizing in caller voice)
                played = await _play_sys_clip(seg_content, audio_stream, skip_flag_fn, gain)
                if not played:
                    # Fallback: synthesize slug text in caller voice
                    readable = seg_content.replace('_', ' ')
                    if not await _synthesize_and_play_clause(readable, is_first=(clause_offset == 0 and not prefetch_consumed)):
                        break
                    clause_offset += 1
                continue

            # Text segment
            seg_clauses = split_clauses(seg_content)

            if not prefetch_consumed and first_chunks:
                # Play the prefetched first clause
                if resume_from_clause <= clause_offset:
                    all_clauses_for_split = split_clauses(seg_content)
                    first_clause_text = all_clauses_for_split[0] if all_clauses_for_split else ""
                    split_char = _get_split_char(first_clause_text)

                    if on_clause_start:
                        on_clause_start(clause_offset, first_clause_text)

                    full_audio = np.concatenate([a.squeeze() for a, sr in first_chunks])
                    trimmed = trim_clause_audio(full_audio, split_char, None, is_first=True)
                    await _play_audio(trimmed)
                    prev_split_char = split_char
                prefetch_consumed = True

                # remaining_clauses from prefetch are already the tail of this segment
                for i, clause in enumerate(remaining_clauses):
                    ci = clause_offset + 1 + i
                    if ci < resume_from_clause:
                        continue
                    if skip_flag_fn():
                        break
                    if on_clause_start:
                        on_clause_start(ci, clause)
                    if not await _synthesize_and_play_clause(clause, is_first=False):
                        break
                clause_offset += len(seg_clauses)
                continue

            # Normal text segment (no prefetch for this one)
            for i, clause in enumerate(seg_clauses):
                ci = clause_offset + i
                if ci < resume_from_clause:
                    continue
                if skip_flag_fn():
                    break
                if on_clause_start:
                    on_clause_start(ci, clause)
                if not await _synthesize_and_play_clause(clause, is_first=(ci == 0)):
                    break
            clause_offset += len(seg_clauses)

    else:
        # No prefetch — iterate segments directly
        first_played = True
        for seg_kind, seg_content in segments:
            if skip_flag_fn():
                break

            if seg_kind == 'sys':
                played = await _play_sys_clip(seg_content, audio_stream, skip_flag_fn, gain)
                if not played:
                    readable = seg_content.replace('_', ' ')
                    if not await _synthesize_and_play_clause(readable, is_first=first_played):
                        break
                    first_played = False
                    clause_offset += 1
                continue

            # Text segment
            seg_clauses = split_clauses(seg_content)
            for i, clause in enumerate(seg_clauses):
                ci = clause_offset + i
                if ci < resume_from_clause:
                    continue
                if skip_flag_fn():
                    break
                if on_clause_start:
                    on_clause_start(ci, clause)
                if not await _synthesize_and_play_clause(clause, is_first=first_played):
                    break
                first_played = False
            clause_offset += len(seg_clauses)

    total_ms = (time.monotonic() - item_t0) * 1000
    proc_alive = audio_stream.is_alive
    print(
        f"speak-daemon: [q#{qid}] DONE   "
        f"total={total_ms:.0f}ms audio={total_audio_secs:.2f}s "
        f"chunks={chunks_done} "
        f"audio={'alive' if proc_alive else 'DEAD'}",
        file=sys.stderr,
    )
    log_event("request_done", qid=qid, total_ms=round(total_ms, 1),
              audio_secs=round(total_audio_secs, 2),
              chunks=chunks_done)
    return resume_from_clause + chunks_done
