"""Speech rendering with clause-level streaming.

Splits text into clauses before synthesis so the first clause can be
prefetched during caller tone playback. Kokoro generates the entire
utterance as a single chunk when phonemes < 510 (most normal text),
so without splitting, prefetch has to wait for the whole thing.

Benchmark data (Kokoro v1.0, Apple M-series):
  tiny  (7 phonemes):   ~350ms
  short (24 phonemes):  ~400ms
  medium(69 phonemes):  ~850ms
  long  (291 phonemes): ~3500ms

By splitting into clauses, the first clause is typically 5-30 phonemes
and can be synthesized in 300-500ms — well within the caller tone
duration (~400-960ms).
"""

import asyncio
import sys
import time
from typing import AsyncIterator

import numpy as np

from .config import SAMPLE_RATE
from .playback_device import AudioOutputStream
from .protocol import log_event
from .text import split_clauses


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

    async def _process_chunk(audio, sr):
        """Process and play a single audio chunk."""
        nonlocal total_audio_secs, chunks_done, chunk_idx

        # create_stream returns 2D array (1, N) — squeeze to 1D
        audio = audio.squeeze()

        # Convert float32 audio to int16 PCM
        pcm_samples = (audio * 32767).astype(np.int16)

        # Apply volume gain
        if gain != 1.0:
            pcm_samples = np.clip(
                pcm_samples.astype(np.float32) * gain, -32767, 32767
            ).astype(np.int16)

        pcm = pcm_samples.tobytes()
        dur = len(pcm_samples) / SAMPLE_RATE

        log_event("chunk_ready", qid=qid, chunk=chunk_idx + 1,
                  audio_secs=round(dur, 2), audio_bytes=len(pcm))

        # Fire timing callback BEFORE write_pcm so it captures when audio
        # is ready, not when playback of the full chunk finishes
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

    async def _synthesize_clause(clause):
        """Synthesize a single clause and play its chunks."""
        async for audio, sr in synth.kokoro.create_stream(
            clause, voice_name, speed, lang, trim=False
        ):
            if skip_flag_fn():
                return False
            await _process_chunk(audio, sr)
        return True

    if prefetch is not None:
        first_chunks, remaining_clauses = prefetch

        # Play prefetched first clause
        for audio, sr in first_chunks:
            if skip_flag_fn():
                break
            await _process_chunk(audio, sr)

        # Synthesize and play remaining clauses
        if not skip_flag_fn():
            for clause in remaining_clauses:
                if skip_flag_fn():
                    print(f"speak-daemon: [q#{qid}] SKIPPED after {chunks_done} chunks", file=sys.stderr)
                    break
                if not await _synthesize_clause(clause):
                    print(f"speak-daemon: [q#{qid}] SKIPPED after {chunks_done} chunks", file=sys.stderr)
                    break
    else:
        # No prefetch: split into clauses ourselves
        clauses = split_clauses(text)
        for clause in clauses:
            if skip_flag_fn():
                print(f"speak-daemon: [q#{qid}] SKIPPED after {chunks_done} chunks", file=sys.stderr)
                break
            if not await _synthesize_clause(clause):
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
