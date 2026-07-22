"""Audio output via sounddevice (PortAudio).

sounddevice.RawOutputStream opens the device synchronously, so when start()
returns, the device IS ready. No silence priming needed.
"""

import asyncio
import sys

import sounddevice as sd

from .config import SAMPLE_RATE

# Write PCM in small chunks so PortAudio backpressure naturally paces us.
WRITE_CHUNK_BYTES = int(SAMPLE_RATE * 2 * 0.25)  # 0.25s of audio per write

# Always-present built-in output. When the configured/system-default device
# can't be opened (classic case: the system default is a disconnected
# Bluetooth headset, which PortAudio still reports as default and then fails to
# open with -9986), we fall back to this so audio never silently dies. Matched
# by substring, so it survives shifting device indices.
FALLBACK_DEVICE = "MacBook Pro Speakers"


class AudioOutputStream:
    """Manages a sounddevice RawOutputStream for continuous PCM streaming."""

    def __init__(self, subscriber_manager=None, device=None):
        self._stream: sd.RawOutputStream | None = None
        self._subscriber_manager = subscriber_manager
        self._device = device  # int index, str name substring, or None for default

    @property
    def is_alive(self) -> bool:
        return self._stream is not None and self._stream.active

    async def set_device(self, device) -> None:
        """Switch to a different audio device. Kills current stream so next write reopens."""
        self._device = device
        await self.kill()

    async def kill(self, force: bool = False) -> None:
        """Shut down the audio stream.

        force=True: abort() — discard buffer, immediate stop (for skip).
        force=False: stop() — drain buffer, then close.
        """
        if self._stream is not None:
            try:
                loop = asyncio.get_event_loop()
                if force:
                    await loop.run_in_executor(None, self._stream.abort)
                else:
                    await loop.run_in_executor(None, self._stream.stop)
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    async def ensure_running(self) -> None:
        if self._stream is not None and self._stream.active:
            return
        # Clean up stale stream
        if self._stream is not None:
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        loop = asyncio.get_event_loop()

        # Try the configured device first (None = system default). If it can't
        # be opened, fall back to the built-in speakers so a dead default
        # device (e.g. a disconnected Bluetooth headset) never silences speak.
        candidates = [self._device]
        if self._device != FALLBACK_DEVICE:
            candidates.append(FALLBACK_DEVICE)

        last_err: Exception | None = None
        for cand in candidates:
            def _open(device=cand):
                s = sd.RawOutputStream(
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    dtype="int16",
                    latency="low",
                    device=device,
                )
                s.start()
                return s

            try:
                self._stream = await loop.run_in_executor(None, _open)
                if cand != self._device:
                    print(
                        f"speak-daemon: audio device {self._device!r} unavailable, "
                        f"fell back to {cand!r}",
                        file=sys.stderr,
                    )
                return
            except Exception as e:  # noqa: BLE001 — try next candidate
                last_err = e

        # Every candidate failed — surface the original error.
        raise last_err

    async def write_pcm(self, pcm: bytes, skip_flag_fn=None) -> float:
        """Write PCM to the audio device in small chunks for backpressure pacing.

        Returns duration in seconds of what was written.

        skip_flag_fn: optional callable returning bool, checked each chunk
        to allow early exit.
        """
        await self.ensure_running()
        loop = asyncio.get_event_loop()
        offset = 0
        while offset < len(pcm):
            if skip_flag_fn and skip_flag_fn():
                break
            chunk = pcm[offset : offset + WRITE_CHUNK_BYTES]
            try:
                await loop.run_in_executor(None, self._stream.write, chunk)
            except sd.PortAudioError as e:
                print(f"speak-daemon: PortAudio error, reopening: {e}", file=sys.stderr)
                try:
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None
                await self.ensure_running()
                await loop.run_in_executor(None, self._stream.write, chunk)
            if self._subscriber_manager:
                self._subscriber_manager.broadcast_audio(chunk)
            offset += len(chunk)
        n_samples = len(pcm) // 2  # int16
        return n_samples / SAMPLE_RATE
