"""Playback queue: FIFO queue for fire-and-forget TTS over a live audio stream.

A single AudioOutputStream (sounddevice/PortAudio) stays open for the lifetime
of the queue, receiving a continuous PCM stream. Items flow seamlessly with no
gaps. The worker uses time-based tracking to know when each item's audio has
finished playing, keeping queue status accurate.

Skip aborts the audio stream (the next write reopens it automatically).
"""

import asyncio
import os
import sys
import time
import traceback
from typing import Callable

from .config import DEFAULT_SPEED
from .playback_device import AudioOutputStream
from .history import SpeechHistory
from .protocol import publish_state
from .renderer import audio_to_pcm, prefetch_first_chunk, render_speech
from .synthesis import SynthesisEngine
from .tones import (
    CALLER_GAP,
    SEPARATOR_TONE,
    get_caller_tone,
)
from .voice_pool import VoicePool

# Per-request latency instrumentation is off by default; set SPEAK_DEBUG_TIMING=1
# to collect the t_* marks and emit the TIMING / prefetch-await diagnostics.
DEBUG_TIMING = os.environ.get("SPEAK_DEBUG_TIMING") == "1"


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
        self._audio = AudioOutputStream(subscriber_manager=subscriber_manager, device=device)
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
        self._paused = False
        self._resume_event = asyncio.Event()
        self._resume_event.set()  # starts unblocked
        self._priority_request = None  # jump-the-line item
        self._paused_request: dict | None = None  # holds item to replay after resume
        self._resume_mid_clause = True  # toggle: resume from interrupted clause vs. start

    def record_history(self, text: str, caller: str = "", session: str = "") -> int:
        return self._history.record(text, caller=caller, session=session)

    def update_history_voice(self, row_id: int, voice: str):
        self._history.update_voice(row_id, voice)

    # --- Public queue introspection (no underscore reach-through for callers) ---

    def pending_count(self) -> int:
        """Number of items waiting in the queue (excludes the item playing)."""
        return self._queue.qsize()

    def current_summary(self) -> str | None:
        """Truncated text of the item currently playing, or None when idle."""
        if self._current is None:
            return None
        return self._current.get("text", "")[:80]

    def current_broadcast(self) -> dict | None:
        """Metadata for the currently playing item, shaped for subscriber events.

        Returns {id, caller, voice, text} or None when nothing is playing.
        """
        if self._current is None:
            return None
        return {
            "id": self._current.get("_queue_id"),
            "caller": self._current.get("caller", ""),
            "voice": self._current.get("_resolved_voice", ""),
            "text": self._current.get("text", "")[:120],
        }

    @property
    def resume_mid_clause(self) -> bool:
        """Whether a paused item resumes from its interrupted clause vs. the start."""
        return self._resume_mid_clause

    @resume_mid_clause.setter
    def resume_mid_clause(self, enabled: bool) -> None:
        self._resume_mid_clause = bool(enabled)

    # --- Public history delegation (server never touches self._history directly) ---

    def history_get(self, n: int = 10, offset: int = 0) -> tuple[list[dict], int]:
        return self._history.get(n, offset)

    def history_by_session(self, session: str, n: int = 10, offset: int = 0) -> tuple[list[dict], int]:
        return self._history.get_by_session(session, n, offset)

    def history_by_caller(self, caller: str, n: int = 10, offset: int = 0) -> tuple[list[dict], int]:
        return self._history.get_by_caller(caller, n, offset)

    def history_by_id(self, row_id: int) -> dict | None:
        return self._history.get_by_id(row_id)

    def last_voice_for_caller(self, caller: str) -> str | None:
        return self._history.last_voice_for_caller(caller)

    async def play_raw_pcm(self, pcm: bytes) -> None:
        """Write raw PCM bytes straight to the audio device (used for tones)."""
        await self._audio.write_pcm(pcm)

    async def replay_by_id(self, row_id: int, from_clause: int | None = None) -> dict:
        """Replay a history entry by ID using its original voice.

        If from_clause is given, playback starts from that clause index.
        """
        entry = self._history.get_by_id(row_id)
        if entry is None:
            return {"ok": False, "error": f"history id {row_id} not found"}
        req = {
            "text": entry["text"],
            "voice": entry["voice"] or "af_heart",
            "speed": DEFAULT_SPEED,
            "lang": "en-us",
            "caller": entry["caller"],
            "session": entry["session"],
            "enqueue": True,
            "_fixed_voice": True,   # bypass voice pool
            "_skip_history": True,  # don't record again
        }
        if from_clause is not None:
            req["_resume_from_clause"] = int(from_clause)
        self._id_counter += 1
        self.total_enqueued += 1
        req["_queue_id"] = self._id_counter
        await self._queue.put(req)
        self._publish("enqueued", enqueued_id=self._id_counter)
        return {"ok": True, "id": row_id, "text": entry["text"][:80], "from_clause": from_clause}

    async def set_device(self, device):
        """Switch audio output device at runtime."""
        await self._audio.set_device(device)

    def start(self):
        self._worker_task = asyncio.create_task(self._worker())

    @property
    def is_active(self) -> bool:
        return self._current is not None or not self._queue.empty() or self._paused

    async def enqueue(self, request: dict, priority: bool = False) -> int:
        self._id_counter += 1
        self.total_enqueued += 1
        request["_queue_id"] = self._id_counter
        if priority:
            # Stash as next-up item (worker checks before queue)
            self._priority_request = request
            # Also put a dummy in the queue to wake the worker if it's
            # blocked on queue.get()
            await self._queue.put({"_priority_wakeup": True})
        else:
            await self._queue.put(request)
        self._publish("enqueued", enqueued_id=self._id_counter)
        return self._queue.qsize()

    @property
    def is_paused(self) -> bool:
        return self._paused

    async def pause(self) -> dict:
        """Pause: block the worker from playing. If mid-utterance, stop and stash for replay."""
        if self._paused:
            return {"ok": True, "already_paused": True}
        self._paused = True
        # Block the worker from pulling the next item
        self._resume_event.clear()
        # If something is currently playing, stop it (will be stashed in finally block)
        if self._current:
            self._skip_flag = True
            await self._audio.kill(force=True)
        self._publish("paused")
        return {"ok": True}

    def resume(self) -> dict:
        """Resume: unblock the worker so the re-enqueued item plays."""
        if not self._paused:
            return {"ok": True, "already_playing": True}
        self._paused = False
        self._resume_event.set()
        self._publish("resumed")
        return {"ok": True}

    async def toggle_pause(self) -> dict:
        """Toggle between paused and playing."""
        if self._paused:
            return self.resume()
        return await self.pause()

    async def skip(self) -> dict:
        """Skip current item by aborting the audio stream. Next write reopens it.

        While paused: discards the stashed replay item but stays paused.
        """
        if self._paused and self._paused_request is not None:
            skipped_text = self._paused_request.get("text", "")[:80]
            self._paused_request = None
            self.total_skipped += 1
            self._publish("skipped")
            return {"ok": True, "skipped": skipped_text}
        if self._current:
            self._skip_flag = True
            self.total_skipped += 1
            await self._audio.kill(force=True)
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
        result = {"pending": len(pending), "items": pending,
                  "paused": self._paused}
        if self._current:
            result["playing"] = {
                "id": self._current.get("_queue_id"),
                "text": self._current.get("text", "")[:80],
                "history_id": self._current.get("_history_id"),
                "clause_idx": self._current.get("_clause_idx"),
                "clause_text": self._current.get("_clause_text", ""),
            }
        elif self._paused and self._paused_request is not None:
            result["playing"] = {
                "id": self._paused_request.get("_queue_id"),
                "text": self._paused_request.get("text", "")[:80],
                "history_id": self._paused_request.get("_history_id"),
                "paused": True,
            }
        return result

    def _publish(self, event: str, **extra):
        """Publish state change for external tools and broadcast to subscribers."""
        pending = list(self._queue._queue)
        state = {
            "event": event,
            "playing": None,
            "pending": len(pending),
            "paused": self._paused,
            "queue": [{"id": r.get("_queue_id"), "caller": r.get("caller", ""),
                       "text": r.get("text", "")[:120]} for r in pending],
        }
        if self._current:
            state["playing"] = {
                "id": self._current.get("_queue_id"),
                "caller": self._current.get("caller", ""),
                "voice": self._current.get("_resolved_voice", ""),
                "text": self._current.get("text", "")[:120],
                "history_id": self._current.get("_history_id"),
                "clause_idx": self._current.get("_clause_idx"),
            }
        state.update(extra)
        publish_state(state)
        if self._subscriber_manager:
            self._subscriber_manager.broadcast_metadata(state)

    async def _worker(self):
        loop = asyncio.get_event_loop()
        self._publish("idle")
        while True:
            # Priority: paused replay > priority enqueue > normal queue
            if self._paused_request is not None:
                # Wait for resume before replaying stashed item
                await self._resume_event.wait()
                request = self._paused_request
                self._paused_request = None
            elif self._priority_request is not None:
                request = self._priority_request
                self._priority_request = None
            else:
                # Wait for resume before pulling from queue, so items stay
                # visible to clear() while paused. Also check again after
                # pulling in case pause was set while waiting on queue.get().
                await self._resume_event.wait()
                request = await self._queue.get()
                if not self._resume_event.is_set():
                    # Paused while we were waiting — stash and loop back
                    self._paused_request = request
                    request["_is_resume"] = True
                    continue
                # Skip wakeup dummies from priority enqueue
                if request.get("_priority_wakeup"):
                    continue
            self._current = request
            self._skip_flag = False
            chunks_done = 0  # tracks clause progress for resume-from-clause
            is_resume = request.get("_is_resume", False)
            skip_history = request.get("_skip_history", False)
            # Record in history early so queries see it before playback finishes
            # (but not on resume or replay-by-id — already recorded)
            text_for_history = request.get("text", "")
            if text_for_history and not is_resume and not skip_history:
                row_id = self.record_history(
                    text_for_history,
                    caller=request.get("caller", ""),
                    session=request.get("session", ""),
                )
                request["_history_id"] = row_id
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
                        await self._audio.write_pcm(CALLER_GAP)
                    elif not caller or caller == self._last_caller:
                        await self._audio.write_pcm(SEPARATOR_TONE)

                # Resolve voice via pool (caller+session) or use request default
                voice_name = request.get("voice", "af_heart")
                session = request.get("session", "")
                is_new_claim = False
                gain = 1.0
                if request.get("_fixed_voice"):
                    # Replay-by-id: use the voice from the request, skip pool
                    pass
                elif caller and self.voice_pool:
                    voice_name, gain, is_new_claim = self.voice_pool.get_voice(
                        caller, session, voice_name
                    )
                request["_resolved_voice"] = voice_name
                request["_gain"] = gain

                # Update history row with resolved voice
                history_id = request.get("_history_id")
                if history_id:
                    self.update_history_voice(history_id, voice_name)

                # Kick off TTS synthesis (prefetch first chunk) concurrently
                # with the caller tone so speech is ready when tone ends.
                text = request.get("text", "").strip()
                speed = request.get("speed", 1.0)
                lang = request.get("lang", "en-us")
                qid = request.get("_queue_id", "?")

                # --- TIMING instrumentation (gated behind SPEAK_DEBUG_TIMING) ---
                # _ts() returns a real timestamp only when timing is enabled, so
                # with it off every t_* mark stays None and no work is collected.
                _ts = time.monotonic if DEBUG_TIMING else (lambda: None)
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
                    t_first_speech_write = _ts()

                def _on_clause_start(idx: int, text: str) -> None:
                    if self._current:
                        self._current["_clause_idx"] = idx
                        self._current["_clause_text"] = text[:80]

                resume_from_clause = request.get("_resume_from_clause", 0)

                prefetch_task = None
                if text and resume_from_clause == 0:
                    # Only prefetch when starting from the beginning;
                    # resuming mid-clause skips the first clause anyway.
                    t_prefetch_start = _ts()
                    prefetch_task = asyncio.create_task(
                        prefetch_first_chunk(self.synth, text, voice_name, speed, lang)
                    )

                # Start tone
                t_tone_start = _ts()
                if caller:
                    await self._audio.write_pcm(get_caller_tone(caller))
                t_tone_done = _ts()

                # Announce new voice assignment
                t_announce_start = _ts()
                if is_new_claim and caller:
                    announce_text = f"{caller} here"
                    async for audio, sr in self.synth.kokoro.create_stream(
                        announce_text, voice_name, DEFAULT_SPEED, "en-us", trim=False
                    ):
                        await self._audio.write_pcm(audio_to_pcm(audio.squeeze(), gain))
                t_announce_done = _ts()

                t_publish_start = _ts()
                self._publish("playing")
                t_publish_done = _ts()

                # Await prefetched first chunk (should be ready by now)
                prefetch = None
                if prefetch_task is not None:
                    t_await_prefetch_start = _ts()
                    prefetch = await prefetch_task
                    t_await_prefetch_done = _ts()
                    # Log whether prefetch finished before or after the tone
                    if DEBUG_TIMING and t_prefetch_start is not None:
                        prefetch_total_ms = (t_await_prefetch_done - t_prefetch_start) * 1000
                        await_cost_ms = (t_await_prefetch_done - t_await_prefetch_start) * 1000
                        print(
                            f"speak-daemon: [q#{qid}] prefetch total={prefetch_total_ms:.0f}ms "
                            f"await_cost={await_cost_ms:.0f}ms",
                            file=sys.stderr,
                        )

                # The actual speech (uses prefetched first chunk if available)
                chunks_done = await render_speech(
                    request, loop, self.synth, self._audio,
                    skip_flag_fn=lambda: self._skip_flag,
                    bg_task_tracker=self._bg_task_tracker,
                    prefetch=prefetch,
                    on_first_write=_on_first_speech_write,
                    resume_from_clause=resume_from_clause,
                    on_clause_start=_on_clause_start,
                )

                # --- TIMING summary ---
                if DEBUG_TIMING:
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
                    await self._audio.write_pcm(get_caller_tone(caller))

                self._publish("item_done")
                self._last_request = request
                self._last_caller = caller
                self._items_played += 1
                self.total_completed += 1
            except Exception as e:
                print(f"speak-daemon: queue playback error: {e}", file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)
            finally:
                # If this was a pause, stash the item for replay on resume
                if self._paused:
                    request["_is_resume"] = True
                    if self._resume_mid_clause:
                        request["_resume_from_clause"] = chunks_done
                        print(
                            f"speak-daemon: paused — will resume from clause {chunks_done}",
                            file=sys.stderr,
                        )
                    else:
                        request.pop("_resume_from_clause", None)
                        print(f"speak-daemon: paused — will replay from start", file=sys.stderr)
                    self._paused_request = request

                self._current = None
                self._on_activity()
                # Reset separator counter when queue drains, and stop the audio
                # stream so the next batch gets a fresh device (handles device changes)
                if self._queue.empty() and not self._paused:
                    self._items_played = 0
                    await self._audio.kill()
                    self._publish("idle")

