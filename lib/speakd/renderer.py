"""Clause-by-clause speech rendering with pipelined synthesis and playback.

Uses a producer/consumer pattern: synthesis runs ahead of playback so
the next clause is ready before the current one finishes playing.
"""

import asyncio
import sys
import time

import numpy as np

from .ffplay import FFPlayStream
from .protocol import log_event
from .synthesis import SynthesisEngine
from .text import split_clauses
from .tones import get_caller_voice

# Pre-synthesize up to 2 clauses ahead of playback
_PIPELINE_DEPTH = 2

_DONE = object()


async def render_speech(
    request: dict,
    loop: asyncio.AbstractEventLoop,
    synth: SynthesisEngine,
    ffplay: FFPlayStream,
    skip_flag_fn,
    bg_task_tracker,
) -> None:
    """Synthesize and stream a full speech request clause by clause.

    Runs synthesis and playback concurrently: while one clause plays
    through ffplay, the next clause(s) are already being synthesized.
    """
    text = request.get("text", "").strip()
    voice_name = request.get("voice", "af_heart")
    speed = request.get("speed", 1.0)
    lang = request.get("lang", "en-us")
    qid = request.get("_queue_id", "?")
    caller = request.get("caller", "")

    gain = 1.0
    if caller:
        voice_name, gain = get_caller_voice(caller, voice_name)
    request["_resolved_voice"] = voice_name

    if not text:
        return

    item_t0 = time.monotonic()
    label = text[:60].replace('\n', ' ')
    caller_tag = f" caller={caller}" if caller else ""
    gain_tag = f" gain={gain}" if gain != 1.0 else ""
    print(f"speak-daemon: [q#{qid}] START  voice={voice_name} speed={speed}{caller_tag}{gain_tag} \"{label}\"", file=sys.stderr)
    log_event("request_start", qid=qid, voice=voice_name, speed=speed,
              caller=caller, text=label)

    voice = synth.kokoro.get_voice_style(voice_name)
    clauses = split_clauses(text)
    n_clauses = len(clauses)

    stats = {"total_synth_ms": 0.0, "total_audio_secs": 0.0, "clauses_done": 0}

    pipe = asyncio.Queue(maxsize=_PIPELINE_DEPTH)

    async def producer():
        """Synthesize clauses and feed PCM into the pipeline queue."""
        for i, sentence in enumerate(clauses):
            if skip_flag_fn():
                break

            synth_t0 = time.monotonic()
            pcm, needs_upgrade = await loop.run_in_executor(
                None, synth.synthesize_sentence,
                sentence, voice_name, voice, speed, lang,
            )
            synth_ms = (time.monotonic() - synth_t0) * 1000
            stats["total_synth_ms"] += synth_ms

            if skip_flag_fn():
                break

            # Apply volume gain
            if gain != 1.0:
                samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
                samples = np.clip(samples * gain, -32767, 32767).astype(np.int16)
                pcm = samples.tobytes()

            cache_tag = "HIT" if synth_ms < 5 else ("ASM" if needs_upgrade else "SYN")
            enqueued_at = time.monotonic()

            log_event("clause_synthesized", qid=qid, clause=i+1, n=n_clauses,
                      synth_ms=round(synth_ms, 1), cache=cache_tag,
                      audio_bytes=len(pcm))

            await pipe.put((i, sentence, pcm, synth_ms, cache_tag, enqueued_at))

            if needs_upgrade:
                future = loop.run_in_executor(
                    None, synth.bg_upgrade,
                    sentence, voice_name, voice, speed, lang,
                )
                bg_task_tracker(future)

        await pipe.put(_DONE)

    async def consumer():
        """Pull synthesized PCM from the queue and write to ffplay."""
        last_write_end = None
        while True:
            item = await pipe.get()
            if item is _DONE:
                break
            if skip_flag_fn():
                continue

            i, sentence, pcm, synth_ms, cache_tag, enqueued_at = item
            dequeued_at = time.monotonic()
            queue_wait_ms = (dequeued_at - enqueued_at) * 1000

            # Gap = time between previous clause's last write and this clause's first write
            gap_ms = (dequeued_at - last_write_end) * 1000 if last_write_end else 0.0

            n_samples = len(pcm) // 2
            audio_dur = n_samples / 24000  # SAMPLE_RATE

            log_event("clause_play_start", qid=qid, clause=i+1, n=n_clauses,
                      queue_wait_ms=round(queue_wait_ms, 1),
                      gap_ms=round(gap_ms, 1), audio_secs=round(audio_dur, 2))

            write_t0 = time.monotonic()
            dur = await ffplay.write_pcm(pcm, skip_flag_fn=skip_flag_fn)
            write_ms = (time.monotonic() - write_t0) * 1000
            last_write_end = time.monotonic()
            stats["total_audio_secs"] += dur
            stats["clauses_done"] += 1

            log_event("clause_play_end", qid=qid, clause=i+1, n=n_clauses,
                      write_ms=round(write_ms, 1), audio_secs=round(dur, 2))

            clause_label = sentence[:40].replace('\n', ' ')
            print(
                f"speak-daemon: [q#{qid}]   clause {i+1}/{n_clauses} "
                f"{cache_tag} synth={synth_ms:.0f}ms write={write_ms:.0f}ms "
                f"gap={gap_ms:.0f}ms wait={queue_wait_ms:.0f}ms "
                f"audio={dur:.2f}s \"{clause_label}\"",
                file=sys.stderr,
            )

    producer_task = asyncio.create_task(producer())
    consumer_task = asyncio.create_task(consumer())
    await asyncio.gather(producer_task, consumer_task)

    total_ms = (time.monotonic() - item_t0) * 1000
    proc_alive = ffplay.is_alive
    done = stats["clauses_done"]
    if done < n_clauses:
        print(f"speak-daemon: [q#{qid}] SKIPPED after {done}/{n_clauses} clauses", file=sys.stderr)
    print(
        f"speak-daemon: [q#{qid}] DONE   "
        f"total={total_ms:.0f}ms audio={stats['total_audio_secs']:.2f}s "
        f"synth={stats['total_synth_ms']:.0f}ms ffplay={'alive' if proc_alive else 'DEAD'}",
        file=sys.stderr,
    )
    log_event("request_done", qid=qid, total_ms=round(total_ms, 1),
              audio_secs=round(stats["total_audio_secs"], 2),
              synth_ms=round(stats["total_synth_ms"], 1),
              clauses_done=done, clauses_total=n_clauses)
