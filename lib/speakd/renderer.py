"""Speech rendering using Kokoro's native streaming.

Feeds full text to Kokoro's create_stream(), which handles phoneme
splitting internally (only at 510+ phoneme boundaries). Each audio
chunk is written directly to ffplay for playback.

Supports prefetching: the first TTS chunk can be generated concurrently
with caller tone playback to eliminate the audible gap between them.
"""

import asyncio
import sys
import time
from typing import AsyncIterator

import numpy as np

from .config import SAMPLE_RATE
from .playback_device import AudioOutputStream
from .protocol import log_event


async def prefetch_first_chunk(synth, text, voice_name, speed, lang):
    """Start the TTS stream and fetch the first audio chunk.

    Returns (stream_iter, first_chunk) where first_chunk is (audio, sr)
    or (stream_iter, None) if the stream is empty.

    This allows the caller to kick off TTS synthesis concurrently with
    other work (e.g. playing a caller tone), so the first speech chunk
    is ready immediately after the tone finishes.
    """
    stream = synth.kokoro.create_stream(text, voice_name, speed, lang, trim=False)
    stream_iter = stream.__aiter__()
    try:
        first_chunk = await stream_iter.__anext__()
    except StopAsyncIteration:
        first_chunk = None
    return stream_iter, first_chunk


async def render_speech(
    request: dict,
    loop: asyncio.AbstractEventLoop,
    synth,
    ffplay: AudioOutputStream,
    skip_flag_fn,
    bg_task_tracker,
    prefetch=None,
) -> None:
    """Stream speech from Kokoro directly to ffplay.

    Uses Kokoro's create_stream() which handles text splitting
    internally, preserving natural prosody across clause boundaries.
    Audio is simultaneously saved to disk for potential replay.

    If prefetch is provided, it should be a (stream_iter, first_chunk)
    tuple from prefetch_first_chunk(). The first chunk is played
    immediately and the remaining chunks are read from stream_iter.
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
        """Process and play a single audio chunk. Returns audio duration."""
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

        await ffplay.write_pcm(pcm, skip_flag_fn=skip_flag_fn)

        total_audio_secs += dur
        chunks_done += 1
        chunk_idx += 1

        chunk_label = text[:40].replace('\n', ' ')
        print(
            f"speak-daemon: [q#{qid}]   chunk {chunk_idx} "
            f"audio={dur:.2f}s \"{chunk_label}\"",
            file=sys.stderr,
        )

    if prefetch is not None:
        # Use prefetched stream: play first chunk, then continue iterator
        stream_iter, first_chunk = prefetch
        if first_chunk is not None and not skip_flag_fn():
            await _process_chunk(*first_chunk)
        # Continue with remaining chunks from the same iterator
        async for audio, sr in stream_iter:
            if skip_flag_fn():
                print(f"speak-daemon: [q#{qid}] SKIPPED after {chunks_done} chunks", file=sys.stderr)
                break
            await _process_chunk(audio, sr)
    else:
        # Normal path: create stream from scratch
        async for audio, sr in synth.kokoro.create_stream(text, voice_name, speed, lang, trim=False):
            if skip_flag_fn():
                print(f"speak-daemon: [q#{qid}] SKIPPED after {chunks_done} chunks", file=sys.stderr)
                break
            await _process_chunk(audio, sr)

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
