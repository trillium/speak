"""Clause-by-clause speech rendering: synthesis, gain, and playback pacing."""

import asyncio
import sys
import time

import numpy as np

from .ffplay import FFPlayStream
from .synthesis import SynthesisEngine
from .text import split_clauses
from .tones import get_caller_voice


async def render_speech(
    request: dict,
    loop: asyncio.AbstractEventLoop,
    synth: SynthesisEngine,
    ffplay: FFPlayStream,
    skip_flag_fn,
    bg_task_tracker,
) -> None:
    """Synthesize and stream a full speech request clause by clause.

    Handles caller voice resolution, per-clause synthesis with cache,
    volume gain, and background cache upgrades.
    """
    text = request.get("text", "").strip()
    voice_name = request.get("voice", "af_heart")
    speed = request.get("speed", 1.0)
    lang = request.get("lang", "en-us")
    qid = request.get("_queue_id", "?")
    caller = request.get("caller", "")

    # Apply caller-specific voice and gain if configured
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

    voice = synth.kokoro.get_voice_style(voice_name)
    clauses = split_clauses(text)
    total_audio_secs = 0
    total_synth_ms = 0

    for i, sentence in enumerate(clauses):
        if skip_flag_fn():
            print(f"speak-daemon: [q#{qid}] SKIPPED after {i}/{len(clauses)} clauses", file=sys.stderr)
            return

        synth_t0 = time.monotonic()
        pcm, needs_upgrade = await loop.run_in_executor(
            None, synth.synthesize_sentence,
            sentence, voice_name, voice, speed, lang,
        )
        synth_ms = (time.monotonic() - synth_t0) * 1000
        total_synth_ms += synth_ms

        if skip_flag_fn():
            print(f"speak-daemon: [q#{qid}] SKIPPED after {i}/{len(clauses)} clauses", file=sys.stderr)
            return

        # Apply volume gain if needed (e.g. quiet voices like af_nova)
        if gain != 1.0:
            samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
            samples = np.clip(samples * gain, -32767, 32767).astype(np.int16)
            pcm = samples.tobytes()

        write_t0 = time.monotonic()
        dur = await ffplay.write_pcm(pcm, skip_flag_fn=skip_flag_fn)
        write_ms = (time.monotonic() - write_t0) * 1000
        total_audio_secs += dur

        cache_tag = "HIT" if synth_ms < 5 else ("ASM" if needs_upgrade else "SYN")
        clause_label = sentence[:40].replace('\n', ' ')
        print(
            f"speak-daemon: [q#{qid}]   clause {i+1}/{len(clauses)} "
            f"{cache_tag} synth={synth_ms:.0f}ms write={write_ms:.0f}ms "
            f"audio={dur:.2f}s speed={speed} \"{clause_label}\"",
            file=sys.stderr,
        )

        if needs_upgrade:
            task = asyncio.create_task(loop.run_in_executor(
                None, synth.bg_upgrade,
                sentence, voice_name, voice, speed, lang,
            ))
            bg_task_tracker(task)

    # No explicit wait needed — backpressure from chunked writes keeps us
    # paced with ffplay. By the time all writes complete, most audio has
    # already played. The persistent stream means the next item flows
    # seamlessly without any gap.
    total_ms = (time.monotonic() - item_t0) * 1000
    proc_alive = ffplay.is_alive
    print(
        f"speak-daemon: [q#{qid}] DONE   "
        f"total={total_ms:.0f}ms audio={total_audio_secs:.2f}s "
        f"synth={total_synth_ms:.0f}ms ffplay={'alive' if proc_alive else 'DEAD'}",
        file=sys.stderr,
    )
