"""Speech rendering with clause-level streaming and configurable trim.

Splits text into clauses before synthesis so the first clause can be
prefetched during caller tone playback. Each clause's audio is trimmed
of Kokoro's built-in silence padding (~280ms lead, ~360ms trail) and
replaced with punctuation-appropriate gaps from config/trim.yaml.

Config is re-read on every synthesis so edits take effect immediately.
"""

import asyncio
import os
import sys
import time
from typing import AsyncIterator

import numpy as np

from .config import SAMPLE_RATE
from .playback_device import AudioOutputStream
from .protocol import log_event
from .text import split_clauses

# Silence detection threshold: fraction of peak amplitude
_SILENCE_THRESH = 0.001

_TRIM_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "config", "trim.yaml"
)


def _load_trim_config():
    """Load trim.yaml, returning (gaps_dict, default_gap_ms)."""
    import yaml

    try:
        with open(_TRIM_CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        gaps = cfg.get("gaps", {})
        default = cfg.get("default_gap", 200)
        return gaps, default
    except (FileNotFoundError, yaml.YAMLError):
        return {}, 200


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


async def prefetch_first_chunk(synth, text, voice_name, speed, lang):
    """Synthesize the first clause of text concurrently with tone playback.

    Splits text into clauses and synthesizes only the first one. Returns
    (first_audio_chunks, remaining_clauses) where first_audio_chunks is
    a list of (audio, sr) tuples from the first clause's stream, and
    remaining_clauses is a list of strings still to be synthesized.

    This keeps prefetch fast (~300-500ms) regardless of total text length.
    """
    t0 = time.monotonic()
    label = text[:40].replace('\n', ' ')

    clauses = split_clauses(text)
    if not clauses:
        print(f"speak-daemon: prefetch_first_chunk EMPTY text=\"{label}\"", file=sys.stderr)
        return [], []

    first_clause = clauses[0]
    remaining = clauses[1:]

    print(
        f"speak-daemon: prefetch_first_chunk STARTED voice={voice_name} "
        f"clauses={len(clauses)} first=\"{first_clause[:40]}\"",
        file=sys.stderr,
    )

    # Synthesize just the first clause
    first_chunks = []
    async for audio, sr in synth.kokoro.create_stream(
        first_clause, voice_name, speed, lang, trim=False
    ):
        first_chunks.append((audio, sr))

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
    ffplay: AudioOutputStream,
    skip_flag_fn,
    bg_task_tracker,
    prefetch=None,
    on_first_write=None,
) -> None:
    """Stream speech from Kokoro to the audio device, clause by clause.

    If prefetch is provided, it should be (first_chunks, remaining_clauses)
    from prefetch_first_chunk(). The first clause audio is played immediately,
    then remaining clauses are synthesized and played sequentially.

    Without prefetch, the full text is split into clauses and each is
    synthesized independently.
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
        return

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
        """Convert float32 audio to PCM and write to device."""
        nonlocal total_audio_secs, chunks_done, chunk_idx

        pcm_samples = (audio * 32767).astype(np.int16)

        if gain != 1.0:
            pcm_samples = np.clip(
                pcm_samples.astype(np.float32) * gain, -32767, 32767
            ).astype(np.int16)

        pcm = pcm_samples.tobytes()
        dur = len(pcm_samples) / SAMPLE_RATE

        log_event("chunk_ready", qid=qid, chunk=chunk_idx + 1,
                  audio_secs=round(dur, 2), audio_bytes=len(pcm))

        if chunk_idx == 0 and on_first_write is not None:
            on_first_write()

        await ffplay.write_pcm(pcm, skip_flag_fn=skip_flag_fn)

        total_audio_secs += dur
        chunks_done += 1
        chunk_idx += 1

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

    if prefetch is not None:
        first_chunks, remaining_clauses = prefetch

        if first_chunks and not skip_flag_fn():
            # Get split char from the first clause text
            all_clauses = split_clauses(text)
            first_clause_text = all_clauses[0] if all_clauses else ""
            split_char = _get_split_char(first_clause_text)

            # Concatenate prefetched audio and trim
            full_audio = np.concatenate([a.squeeze() for a, sr in first_chunks])
            trimmed = trim_clause_audio(full_audio, split_char, None, is_first=True)
            await _play_audio(trimmed)
            prev_split_char = split_char

        # Synthesize remaining clauses
        if not skip_flag_fn():
            for i, clause in enumerate(remaining_clauses):
                if skip_flag_fn():
                    print(f"speak-daemon: [q#{qid}] SKIPPED after {chunks_done} chunks", file=sys.stderr)
                    break
                if not await _synthesize_and_play_clause(clause, is_first=False):
                    print(f"speak-daemon: [q#{qid}] SKIPPED after {chunks_done} chunks", file=sys.stderr)
                    break
    else:
        clauses = split_clauses(text)
        for i, clause in enumerate(clauses):
            if skip_flag_fn():
                print(f"speak-daemon: [q#{qid}] SKIPPED after {chunks_done} chunks", file=sys.stderr)
                break
            if not await _synthesize_and_play_clause(clause, is_first=(i == 0)):
                print(f"speak-daemon: [q#{qid}] SKIPPED after {chunks_done} chunks", file=sys.stderr)
                break

    total_ms = (time.monotonic() - item_t0) * 1000
    proc_alive = ffplay.is_alive
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
