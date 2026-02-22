"""SpeakDaemon server: Unix socket listener, client handler, and main entry point."""

import asyncio
import json
import os
import signal
import struct
import sys
import time

from kokoro_onnx import Kokoro

from .cache import AudioCache
from .config import CACHE_DIR, CACHE_TTL_DAYS, IDLE_TIMEOUT, SOCKET_PATH
from .kokoro_patch import apply_patch
from .playback import PlaybackQueue
from .protocol import send_json
from .synthesis import SynthesisEngine
from .text import split_clauses
from .voice_pool import VoicePool


class SpeakDaemon:
    def __init__(self, model_path: str, voices_path: str):
        self.kokoro = Kokoro(model_path, voices_path)
        self.cache = AudioCache(CACHE_DIR, CACHE_TTL_DAYS)
        self.synth = SynthesisEngine(self.kokoro, self.cache)
        self.last_activity = time.monotonic()
        self.active_connections = 0
        self._bg_tasks: set[asyncio.Future] = set()
        self.start_time = time.time()
        config_dir = os.path.join(os.path.dirname(__file__), "..", "..", "config")
        voice_config = os.path.join(config_dir, "voices.json")
        self.voice_pool = VoicePool(voice_config)
        self.playback_queue = PlaybackQueue(
            synth=self.synth,
            on_activity=self._touch_activity,
            bg_task_tracker=self._track_bg_task,
            voice_pool=self.voice_pool,
        )

    def _touch_activity(self):
        self.last_activity = time.monotonic()

    def _track_bg_task(self, future: asyncio.Future):
        self._bg_tasks.add(future)
        future.add_done_callback(self._bg_tasks.discard)

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        self.active_connections += 1
        self.last_activity = time.monotonic()
        try:
            # Read length-prefixed JSON request
            raw_len = await reader.readexactly(4)
            msg_len = struct.unpack("!I", raw_len)[0]
            raw_msg = await reader.readexactly(msg_len)
            request = json.loads(raw_msg.decode())

            # --- Queue command dispatch (skip, clear, queue_status) ---
            command = request.get("command")
            if command:
                if command == "skip":
                    result = await self.playback_queue.skip()
                elif command == "clear":
                    result = await self.playback_queue.clear()
                elif command == "queue_status":
                    result = self.playback_queue.status()
                elif command == "replay":
                    result = await self.playback_queue.replay()
                elif command == "stats":
                    q = self.playback_queue
                    result = {
                        "daemon": {
                            "pid": os.getpid(),
                            "uptime_secs": round(time.time() - self.start_time),
                            "active_connections": self.active_connections,
                        },
                        "queue": {
                            "total_enqueued": q.total_enqueued,
                            "total_completed": q.total_completed,
                            "total_skipped": q.total_skipped,
                            "pending": q._queue.qsize(),
                            "playing": q._current.get("text", "")[:80] if q._current else None,
                        },
                        "cache": self.cache.stats(),
                    }
                elif command == "voice_pool_status":
                    result = {"ok": True, **self.voice_pool.status()}
                elif command == "history":
                    n = request.get("n", 10)
                    result = {"ok": True, "entries": self.playback_queue.get_history(n)}
                else:
                    result = {"ok": False, "error": f"unknown command: {command}"}
                send_json(writer, result)
                await writer.drain()
                return

            # --- Enqueue dispatch (fire-and-forget) ---
            if request.get("enqueue"):
                position = await self.playback_queue.enqueue(request)
                send_json(writer, {"ok": True, "position": position})
                await writer.drain()
                return

            # --- Original streaming path (unchanged) ---
            text = request.get("text", "").strip()
            voice_name = request.get("voice", "af_heart")
            speed = request.get("speed", 1.0)
            lang = request.get("lang", "en-us")

            if not text:
                writer.close()
                await writer.wait_closed()
                return

            # Resolve voice style vector once
            voice = self.kokoro.get_voice_style(voice_name)

            # Split into clauses and stream each one as it's ready
            clauses = split_clauses(text)
            loop = asyncio.get_event_loop()

            for sentence in clauses:
                pcm, needs_upgrade = await loop.run_in_executor(
                    None, self.synth.synthesize_sentence,
                    sentence, voice_name, voice, speed, lang,
                )
                writer.write(struct.pack("!I", len(pcm)))
                writer.write(pcm)
                await writer.drain()

                # If we served from word cache, upgrade to clause cache in background
                if needs_upgrade:
                    task = asyncio.create_task(loop.run_in_executor(
                        None, self.synth.bg_upgrade,
                        sentence, voice_name, voice, speed, lang,
                    ))
                    self._track_bg_task(task)

            # Signal end of stream with zero-length chunk
            writer.write(struct.pack("!I", 0))
            await writer.drain()

        except (asyncio.IncompleteReadError, ConnectionResetError, BrokenPipeError):
            pass
        finally:
            self.active_connections -= 1
            self.last_activity = time.monotonic()
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def idle_watchdog(self):
        """Shut down if idle for IDLE_TIMEOUT seconds. Evict expired cache periodically."""
        evict_interval = 3600  # check once per hour
        last_evict = time.monotonic()
        while True:
            await asyncio.sleep(30)
            idle_for = time.monotonic() - self.last_activity
            if self.active_connections == 0 and idle_for >= IDLE_TIMEOUT and not self.playback_queue.is_active:
                print(f"speak-daemon: idle for {IDLE_TIMEOUT}s, shutting down", file=sys.stderr)
                cleanup_and_exit()
            if time.monotonic() - last_evict > evict_interval:
                removed = self.cache.evict_expired()
                if removed:
                    print(f"speak-daemon: evicted {removed} expired cache entries", file=sys.stderr)
                last_evict = time.monotonic()

    async def run(self):
        # Clean up stale socket
        if os.path.exists(SOCKET_PATH):
            os.unlink(SOCKET_PATH)

        server = await asyncio.start_unix_server(self.handle_client, path=SOCKET_PATH)
        os.chmod(SOCKET_PATH, 0o600)

        # Write PID file for management
        pid_path = SOCKET_PATH + ".pid"
        with open(pid_path, "w") as f:
            f.write(str(os.getpid()))

        s = self.cache.stats()
        print(
            f"speak-daemon: listening on {SOCKET_PATH} (pid {os.getpid()})\n"
            f"  cache: {s['clauses']} clauses ({s['clause_hits']} hits), "
            f"{s['words']} words ({s['word_hits']} hits), TTL={CACHE_TTL_DAYS}d",
            file=sys.stderr,
        )

        self.playback_queue.start()
        asyncio.create_task(self.idle_watchdog())

        async with server:
            await server.serve_forever()


def cleanup_and_exit():
    try:
        os.unlink(SOCKET_PATH)
    except FileNotFoundError:
        pass
    try:
        os.unlink(SOCKET_PATH + ".pid")
    except FileNotFoundError:
        pass
    sys.exit(0)


def main():
    import argparse

    # Apply Kokoro speed bug monkeypatch before anything else
    apply_patch()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--voices", required=True)
    args = parser.parse_args()

    # Handle signals for clean shutdown
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: cleanup_and_exit())

    print("speak-daemon: loading model...", file=sys.stderr)
    daemon = SpeakDaemon(args.model, args.voices)
    s = daemon.cache.stats()
    print(
        f"speak-daemon: model loaded, ready. cache={CACHE_DIR}\n"
        f"  {s['clauses']} clauses ({s['clause_hits']} hits), "
        f"{s['words']} words ({s['word_hits']} hits), TTL={CACHE_TTL_DAYS}d",
        file=sys.stderr,
    )
    if s["voices"]:
        for v, vs in sorted(s["voices"].items()):
            print(f"    {v}: {vs['clauses']}c/{vs['words']}w, {vs['hits']} hits", file=sys.stderr)

    asyncio.run(daemon.run())
