"""Playback queue: FIFO queue for fire-and-forget TTS with persistent ffplay.

One ffplay process stays open for the lifetime of the queue, receiving a
continuous PCM stream. Items flow seamlessly with no gaps. The worker uses
time-based tracking to know when each item's audio has finished playing,
keeping queue status accurate.

Skip kills the ffplay process (next write auto-restarts it).
"""

import asyncio
import sys
from typing import Callable

from .playback_device import AudioOutputStream
from .history import SpeechHistory
from .protocol import publish_state
from .renderer import render_speech
from .synthesis import SynthesisEngine
from .tones import (
    CALLER_GAP,
    SEPARATOR_TONE,
    get_caller_tone,
    get_caller_voice,
)


class PlaybackQueue:
    """FIFO queue for fire-and-forget TTS with a single persistent play process."""

    def __init__(
        self,
        synth: SynthesisEngine,
        on_activity: Callable[[], None],
        bg_task_tracker: Callable[[asyncio.Task], None],
    ):
        self.synth = synth
        self._on_activity = on_activity
        self._bg_task_tracker = bg_task_tracker
        self._queue: asyncio.Queue = asyncio.Queue()
        self._current: dict | None = None
        self._ffplay = AudioOutputStream()
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

    def record_history(self, text: str):
        self._history.record(text)

    def get_history(self, n: int = 10) -> list[str]:
        return self._history.get(n)

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
        """Publish state change for external tools."""
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

    async def _worker(self):
        loop = asyncio.get_event_loop()
        self._publish("idle")
        while True:
            request = await self._queue.get()
            self._current = request
            self._skip_flag = False
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

                # Resolve voice early so state events include it
                voice_name = request.get("voice", "af_heart")
                if caller:
                    voice_name, _ = get_caller_voice(caller, voice_name)
                request["_resolved_voice"] = voice_name

                # Start tone
                if caller:
                    await self._ffplay.write_pcm(get_caller_tone(caller))

                self._publish("playing")

                # The actual speech
                await render_speech(
                    request, loop, self.synth, self._ffplay,
                    skip_flag_fn=lambda: self._skip_flag,
                    bg_task_tracker=self._bg_task_tracker,
                )

                # End tone
                if caller:
                    await self._ffplay.write_pcm(get_caller_tone(caller))

                self._publish("item_done")
                self._last_request = request
                self._last_caller = caller
                self._items_played += 1
                self.total_completed += 1
                # Record in history
                text = request.get("text", "")
                if text:
                    self.record_history(text)
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

