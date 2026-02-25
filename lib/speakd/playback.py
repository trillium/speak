"""Playback queue: FIFO queue for fire-and-forget TTS with persistent ffplay.

One ffplay process stays open for the lifetime of the queue, receiving a
continuous PCM stream. Items flow seamlessly with no gaps. The worker uses
time-based tracking to know when each item's audio has finished playing,
keeping queue status accurate.

Skip kills the ffplay process (next write auto-restarts it).
"""

import asyncio
import sys
import time
from typing import Callable

from .playback_device import AudioOutputStream
from .history import SpeechHistory
from .protocol import publish_state
from .renderer import prefetch_first_chunk, render_speech
from .synthesis import SynthesisEngine
from .tones import (
    CALLER_GAP,
    SEPARATOR_TONE,
    get_caller_tone,
)
from .voice_pool import VoicePool


class PlaybackQueue:
    """FIFO queue for fire-and-forget TTS with a single persistent play process."""

    def __init__(
        self,
        synth: SynthesisEngine,
        on_activity: Callable[[], None],
        bg_task_tracker: Callable[[asyncio.Task], None],
        voice_pool: VoicePool | None = None,
        subscriber_manager=None,
        device=None,
    ):
        self.synth = synth
        self._on_activity = on_activity
        self._bg_task_tracker = bg_task_tracker
        self.voice_pool = voice_pool
        self._subscriber_manager = subscriber_manager
        self._queue: asyncio.Queue = asyncio.Queue()
        self._current: dict | None = None
        self._ffplay = AudioOutputStream(subscriber_manager=subscriber_manager, device=device)
        self._worker_task: asyncio.Task | None = None
        self._id_counter = 0
        self._skip_flag = False
        self._last_request: dict | None = None  # for replay
        self._items_played = 0  # track consecutive items for separator tone
        self._last_caller: str | None = None  # track caller for caller-specific tones
        self.total_enqueued = 0
        self.total_completed = 0
        self.total_skipped = 0
        self._history = SpeechHistory()

    def record_history(self, text: str, caller: str = "", session: str = ""):
        self._history.record(text, caller=caller, session=session)

    def get_history(self, n: int = 10) -> list[str]:
        return self._history.get(n)

    def get_history_by_session(self, session: str, n: int = 10) -> list[str]:
        return self._history.get_by_session(session, n)

    def get_history_by_caller(self, caller: str, n: int = 10) -> list[str]:
        return self._history.get_by_caller(caller, n)

    async def set_device(self, device):
        """Switch audio output device at runtime."""
        await self._ffplay.set_device(device)

    def start(self):
        self._worker_task = asyncio.create_task(self._worker())

    @property
    def is_active(self) -> bool:
        return self._current is not None or not self._queue.empty()

    async def enqueue(self, request: dict) -> int:
        self._id_counter += 1
        self.total_enqueued += 1
        request["_queue_id"] = self._id_counter
        await self._queue.put(request)
        self._publish("enqueued", enqueued_id=self._id_counter)
        return self._queue.qsize()

    async def skip(self) -> dict:
        """Skip current item by killing ffplay. Next write restarts it."""
        if self._current:
            self._skip_flag = True
            self.total_skipped += 1
            await self._ffplay.kill(force=True)
            self._publish("skipped")
            return {"ok": True, "skipped": self._current.get("text", "")[:80]}
        return {"ok": False, "error": "nothing playing"}

    async def clear(self) -> dict:
        count = 0
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                count += 1
            except asyncio.QueueEmpty:
                break
        self._publish("cleared", cleared_count=count)
        return {"ok": True, "cleared": count}

    async def replay(self) -> dict:
        """Re-enqueue the last completed item."""
        if self._last_request is None:
            return {"ok": False, "error": "nothing to replay"}
        req = dict(self._last_request)
        self._id_counter += 1
        req["_queue_id"] = self._id_counter
        req["_is_replay"] = True
        await self._queue.put(req)
        return {"ok": True, "position": self._queue.qsize(),
                "text": req.get("text", "")[:80]}

    def status(self) -> dict:
        pending = []
        items = list(self._queue._queue)
        for item in items:
            pending.append({
                "id": item.get("_queue_id"),
                "text": item.get("text", "")[:80],
            })
        result = {"pending": len(pending), "items": pending}
        if self._current:
            result["playing"] = {
                "id": self._current.get("_queue_id"),
                "text": self._current.get("text", "")[:80],
            }
        return result

    def _publish(self, event: str, **extra):
        """Publish state change for external tools and broadcast to subscribers."""
        pending = list(self._queue._queue)
        state = {
            "event": event,
            "playing": None,
            "pending": len(pending),
            "queue": [{"id": r.get("_queue_id"), "caller": r.get("caller", ""),
                       "text": r.get("text", "")[:120]} for r in pending],
        }
        if self._current:
            state["playing"] = {
                "id": self._current.get("_queue_id"),
                "caller": self._current.get("caller", ""),
                "voice": self._current.get("_resolved_voice", ""),
                "text": self._current.get("text", "")[:120],
            }
        state.update(extra)
        publish_state(state)
        if self._subscriber_manager:
            self._subscriber_manager.broadcast_metadata(state)

    async def _worker(self):
        loop = asyncio.get_event_loop()
        self._publish("idle")
        while True:
            request = await self._queue.get()
            self._current = request
            self._skip_flag = False
            # Record in history early so queries see it before playback finishes
            text_for_history = request.get("text", "")
            if text_for_history:
                self.record_history(
                    text_for_history,
                    caller=request.get("caller", ""),
                    session=request.get("session", ""),
                )
            try:
                caller = request.get("caller")
                # Spacing between items:
                #   same caller:  [separator_tone]
                #   diff caller:  [silence gap]  (prev end tone already played)
                #   no caller:    [separator_tone]
                if self._items_played > 0:
                    if caller and caller != self._last_caller:
                        # Gap after previous caller's end tone
                        print(f"speak-daemon: [gap] 1.0s silence between {self._last_caller} -> {caller}", file=sys.stderr)
                        await self._ffplay.write_pcm(CALLER_GAP)
                    elif not caller or caller == self._last_caller:
                        await self._ffplay.write_pcm(SEPARATOR_TONE)

                # Resolve voice via pool (caller+session) or use request default
                voice_name = request.get("voice", "af_heart")
                session = request.get("session", "")
                is_new_claim = False
                gain = 1.0
                if caller and self.voice_pool:
                    voice_name, gain, is_new_claim = self.voice_pool.get_voice(
                        caller, session, voice_name
                    )
                request["_resolved_voice"] = voice_name
                request["_gain"] = gain

                # Kick off TTS synthesis (prefetch first chunk) concurrently
                # with the caller tone so speech is ready when tone ends.
                text = request.get("text", "").strip()
                speed = request.get("speed", 1.0)
                lang = request.get("lang", "en-us")
                qid = request.get("_queue_id", "?")

                # --- TIMING instrumentation ---
                t_prefetch_start = None
                t_tone_start = None
                t_tone_done = None
                t_announce_start = None
                t_announce_done = None
                t_publish_start = None
                t_publish_done = None
                t_await_prefetch_start = None
                t_await_prefetch_done = None
                # This gets set by render_speech via callback
                t_first_speech_write = None

                def _on_first_speech_write():
                    nonlocal t_first_speech_write
                    t_first_speech_write = time.monotonic()

                prefetch_task = None
                if text:
                    t_prefetch_start = time.monotonic()
                    prefetch_task = asyncio.create_task(
                        prefetch_first_chunk(self.synth, text, voice_name, speed, lang)
                    )

                # Start tone
                t_tone_start = time.monotonic()
                if caller:
                    await self._ffplay.write_pcm(get_caller_tone(caller))
                t_tone_done = time.monotonic()

                # Announce new voice assignment
                t_announce_start = time.monotonic()
                if is_new_claim and caller:
                    import numpy as np
                    announce_text = f"{caller} here"
                    async for audio, sr in self.synth.kokoro.create_stream(
                        announce_text, voice_name, 1.26, "en-us", trim=False
                    ):
                        audio = audio.squeeze()
                        pcm_samples = (audio * 32767).astype(np.int16)
                        if gain != 1.0:
                            pcm_samples = np.clip(
                                pcm_samples.astype(np.float32) * gain, -32767, 32767
                            ).astype(np.int16)
                        await self._ffplay.write_pcm(pcm_samples.tobytes())
                t_announce_done = time.monotonic()

                t_publish_start = time.monotonic()
                self._publish("playing")
                t_publish_done = time.monotonic()

                # Await prefetched first chunk (should be ready by now)
                prefetch = None
                if prefetch_task is not None:
                    t_await_prefetch_start = time.monotonic()
                    prefetch = await prefetch_task
                    t_await_prefetch_done = time.monotonic()
                    # Log whether prefetch finished before or after the tone
                    if t_prefetch_start is not None:
                        prefetch_total_ms = (t_await_prefetch_done - t_prefetch_start) * 1000
                        await_cost_ms = (t_await_prefetch_done - t_await_prefetch_start) * 1000
                        print(
                            f"speak-daemon: [q#{qid}] prefetch total={prefetch_total_ms:.0f}ms "
                            f"await_cost={await_cost_ms:.0f}ms",
                            file=sys.stderr,
                        )

                # The actual speech (uses prefetched first chunk if available)
                await render_speech(
                    request, loop, self.synth, self._ffplay,
                    skip_flag_fn=lambda: self._skip_flag,
                    bg_task_tracker=self._bg_task_tracker,
                    prefetch=prefetch,
                    on_first_write=_on_first_speech_write,
                )

                # --- TIMING summary ---
                _ms = lambda a, b: (b - a) * 1000 if (a is not None and b is not None) else 0
                tone_ms = _ms(t_tone_start, t_tone_done)
                announce_ms = _ms(t_announce_start, t_announce_done)
                publish_ms = _ms(t_publish_start, t_publish_done)
                prefetch_ms = _ms(t_prefetch_start, t_await_prefetch_done) if t_prefetch_start and t_await_prefetch_done else 0
                await_prefetch_ms = _ms(t_await_prefetch_start, t_await_prefetch_done)
                gap_ms = _ms(t_tone_done, t_first_speech_write) if t_tone_done and t_first_speech_write else 0
                first_speech_ms = _ms(t_publish_done, t_first_speech_write) if t_publish_done and t_first_speech_write else 0
                print(
                    f"speak-daemon: [q#{qid}] TIMING "
                    f"tone={tone_ms:.0f}ms "
                    f"prefetch={prefetch_ms:.0f}ms "
                    f"await_prefetch={await_prefetch_ms:.0f}ms "
                    f"announce={announce_ms:.0f}ms "
                    f"publish={publish_ms:.0f}ms "
                    f"first_speech={first_speech_ms:.0f}ms "
                    f"gap(tone->speech)={gap_ms:.0f}ms",
                    file=sys.stderr,
                )

                # End tone
                if caller:
                    await self._ffplay.write_pcm(get_caller_tone(caller))

                self._publish("item_done")
                self._last_request = request
                self._last_caller = caller
                self._items_played += 1
                self.total_completed += 1
            except Exception as e:
                print(f"speak-daemon: queue playback error: {e}", file=sys.stderr)
            finally:
                self._current = None
                self._on_activity()
                # Reset separator counter when queue drains, and kill ffplay
                # so next batch gets a fresh audio device (handles device changes)
                if self._queue.empty():
                    self._items_played = 0
                    await self._ffplay.kill()
                    self._publish("idle")

