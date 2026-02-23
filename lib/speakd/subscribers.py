"""Broadcast subscriber management for audio and metadata streaming.

Clients subscribe via the daemon socket and receive a copy of every PCM
chunk written to the audio device, plus optional metadata events. Each
subscriber gets a bounded asyncio queue with a dedicated sender task so
slow clients never block playback.
"""

import asyncio
import json
import sys
import time
from dataclasses import dataclass, field

from .protocol import encode_broadcast_frame, FRAME_TYPE_AUDIO, FRAME_TYPE_METADATA


@dataclass
class SubscriberInfo:
    writer: asyncio.StreamWriter
    include_metadata: bool = True
    connected_at: float = field(default_factory=time.monotonic)
    bytes_sent: int = 0
    dropped_frames: int = 0
    queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=64))
    sender_task: asyncio.Task | None = None
    disconnect_event: asyncio.Event = field(default_factory=asyncio.Event)


class SubscriberManager:
    """Manages broadcast subscribers with per-subscriber send queues."""

    def __init__(self):
        self._subscribers: dict[asyncio.StreamWriter, SubscriberInfo] = {}

    @property
    def count(self) -> int:
        return len(self._subscribers)

    def add(self, writer: asyncio.StreamWriter, include_metadata: bool = True) -> SubscriberInfo:
        info = SubscriberInfo(writer=writer, include_metadata=include_metadata)
        info.sender_task = asyncio.create_task(self._sender(info))
        self._subscribers[writer] = info
        print(f"speak-daemon: subscriber added (total: {self.count})", file=sys.stderr)
        return info

    def remove(self, writer: asyncio.StreamWriter) -> None:
        info = self._subscribers.pop(writer, None)
        if info is None:
            return
        info.disconnect_event.set()
        if info.sender_task and not info.sender_task.done():
            info.sender_task.cancel()
        print(
            f"speak-daemon: subscriber removed (total: {self.count}, "
            f"sent: {info.bytes_sent}, dropped: {info.dropped_frames})",
            file=sys.stderr,
        )

    def broadcast_audio(self, pcm: bytes) -> None:
        """Broadcast raw PCM to all subscribers. Non-blocking."""
        if not self._subscribers:
            return
        frame = encode_broadcast_frame(FRAME_TYPE_AUDIO, pcm)
        for info in list(self._subscribers.values()):
            self._enqueue_frame(info, frame)

    def broadcast_metadata(self, event: dict) -> None:
        """Broadcast a metadata event to subscribers that opted in. Non-blocking."""
        if not self._subscribers:
            return
        payload = json.dumps(event).encode()
        frame = encode_broadcast_frame(FRAME_TYPE_METADATA, payload)
        for info in list(self._subscribers.values()):
            if info.include_metadata:
                self._enqueue_frame(info, frame)

    def _enqueue_frame(self, info: SubscriberInfo, frame: bytes) -> None:
        try:
            info.queue.put_nowait(frame)
        except asyncio.QueueFull:
            # Drop oldest frame to make room
            try:
                info.queue.get_nowait()
                info.dropped_frames += 1
            except asyncio.QueueEmpty:
                pass
            try:
                info.queue.put_nowait(frame)
            except asyncio.QueueFull:
                pass

    async def _sender(self, info: SubscriberInfo) -> None:
        """Per-subscriber coroutine that drains the queue to the socket."""
        try:
            while True:
                frame = await info.queue.get()
                info.writer.write(frame)
                await info.writer.drain()
                info.bytes_sent += len(frame)
        except (ConnectionResetError, BrokenPipeError, ConnectionError, OSError):
            pass
        except asyncio.CancelledError:
            return
        finally:
            self.remove(info.writer)

    async def shutdown(self) -> None:
        """Send zero-length terminator to all subscribers and clean up."""
        terminator = b"\x00\x00\x00\x00"
        for info in list(self._subscribers.values()):
            try:
                info.writer.write(terminator)
                await info.writer.drain()
            except Exception:
                pass
        for writer in list(self._subscribers):
            self.remove(writer)

    def status(self) -> dict:
        return {
            "subscribers": self.count,
            "details": [
                {
                    "connected_secs": round(time.monotonic() - info.connected_at),
                    "bytes_sent": info.bytes_sent,
                    "dropped_frames": info.dropped_frames,
                    "queue_depth": info.queue.qsize(),
                    "include_metadata": info.include_metadata,
                }
                for info in self._subscribers.values()
            ],
        }
