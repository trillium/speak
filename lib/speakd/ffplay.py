"""ffplay process management for PCM audio playback."""

import asyncio

from .config import SAMPLE_RATE

# Write PCM in small chunks so ffplay backpressure naturally paces us.
# When ffplay's internal buffer (~5s) is full, drain() blocks until it
# consumes some audio. This keeps us in sync with actual playback.
WRITE_CHUNK_BYTES = int(SAMPLE_RATE * 2 * 0.25)  # 0.25s of audio per write


class FFPlayStream:
    """Manages a persistent ffplay subprocess for continuous PCM streaming."""

    def __init__(self):
        self._proc: asyncio.subprocess.Process | None = None

    @property
    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    async def kill(self, force: bool = False) -> None:
        """Shut down ffplay. By default, closes stdin and waits for ffplay
        to finish playing its buffer (-autoexit handles this). Use force=True
        to kill immediately (e.g. on skip)."""
        if self._proc and self._proc.returncode is None:
            try:
                if force:
                    self._proc.kill()
                else:
                    self._proc.stdin.close()
                await self._proc.wait()
            except Exception:
                pass
            self._proc = None

    async def ensure_running(self) -> None:
        if self._proc and self._proc.returncode is None:
            return
        self._proc = await asyncio.create_subprocess_exec(
            "ffplay", "-nodisp", "-autoexit",
            "-probesize", "32",
            "-f", "s16le", "-ar", "24000", "-ch_layout", "mono", "-i", "pipe:0",
            "-loglevel", "quiet",
            stdin=asyncio.subprocess.PIPE,
        )
        # Prime ffplay with silence so it finishes probing before real audio
        prime = b"\x00" * (SAMPLE_RATE * 2 // 10)  # 100ms silence
        self._proc.stdin.write(prime)
        await self._proc.stdin.drain()

    async def write_pcm(self, pcm: bytes, skip_flag_fn=None) -> float:
        """Write PCM to ffplay in small chunks for backpressure pacing.
        Returns duration in seconds of what was written.

        skip_flag_fn: optional callable returning bool, checked each chunk to allow early exit.
        """
        await self.ensure_running()
        offset = 0
        while offset < len(pcm):
            if skip_flag_fn and skip_flag_fn():
                break
            chunk = pcm[offset:offset + WRITE_CHUNK_BYTES]
            try:
                self._proc.stdin.write(chunk)
                await self._proc.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                self._proc = None
                await self.ensure_running()
                self._proc.stdin.write(chunk)
                await self._proc.stdin.drain()
            offset += len(chunk)
        n_samples = len(pcm) // 2  # int16
        return n_samples / SAMPLE_RATE
