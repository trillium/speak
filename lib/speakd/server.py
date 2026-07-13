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
from .config import CACHE_DIR, CACHE_TTL_DAYS, DEFAULT_SPEED, IDLE_TIMEOUT, SOCKET_PATH
from .kokoro_patch import apply_patch
from .playback import PlaybackQueue
from .protocol import send_json
from .subscribers import SubscriberManager
from .synthesis import SynthesisEngine
from .voice_pool import VoicePool


class SpeakDaemon:
    def __init__(self, model_path: str, voices_path: str, device=None):
        self.kokoro = Kokoro(model_path, voices_path)
        self.cache = AudioCache(CACHE_DIR, CACHE_TTL_DAYS)
        self.synth = SynthesisEngine(self.kokoro, self.cache)
        self.last_activity = time.monotonic()
        self.active_connections = 0
        self._bg_tasks: set[asyncio.Future] = set()
        self._shutdown: asyncio.Event | None = None
        self.start_time = time.time()
        config_dir = os.path.join(os.path.dirname(__file__), "..", "..", "config")
        voice_config = os.path.join(config_dir, "voices.json")
        self.voice_pool = VoicePool(voice_config)
        self.subscriber_manager = SubscriberManager()
        self.playback_queue = PlaybackQueue(
            synth=self.synth,
            on_activity=self._touch_activity,
            bg_task_tracker=self._track_bg_task,
            voice_pool=self.voice_pool,
            subscriber_manager=self.subscriber_manager,
            device=device,
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

            # --- Command dispatch via module-level registry (see COMMANDS) ---
            command = request.get("command")
            if command:
                handler = COMMANDS.get(command)
                if handler is None:
                    send_json(writer, {"ok": False, "error": f"unknown command: {command}"})
                    await writer.drain()
                    return
                result = await handler(self, request, writer)
                if result is _HANDLED:
                    # Handler owns the connection lifecycle (e.g. subscribe holds
                    # it open); it has already replied and must not be closed here.
                    return
                send_json(writer, result)
                await writer.drain()
                return

            # --- Enqueue dispatch (fire-and-forget) ---
            if request.get("enqueue"):
                priority = request.get("priority", False)
                position = await self.playback_queue.enqueue(request, priority=priority)
                send_json(writer, {"ok": True, "position": position})
                await writer.drain()
                return

            # No command and no enqueue flag. The legacy raw-text streaming
            # path (synthesize clause-by-clause straight to the socket, consumed
            # by speak-client | ffplay) was retired (task-5p7) — enqueue is now
            # the only playback route. Reply once with an error and close rather
            # than leaving the client hanging on a stream that will never come.
            send_json(writer, {"ok": False, "error": "raw streaming path removed; use enqueue"})
            await writer.drain()

        except (asyncio.IncompleteReadError, ConnectionResetError, BrokenPipeError):
            pass
        finally:
            self.active_connections -= 1
            self.last_activity = time.monotonic()
            # Don't close writer if it's a managed subscriber
            if writer not in self.subscriber_manager._subscribers:
                try:
                    writer.close()
                    await writer.wait_closed()
                except Exception:
                    pass

    async def idle_watchdog(self):
        """Evict expired cache periodically. Idle-exit only when unsupervised.

        Under launchd KeepAlive (SPEAK_SUPERVISED=1) the daemon must never
        idle-exit: launchd would immediately respawn it, churning the ~10-15s
        model load on every idle window. Cache eviction still runs regardless.
        """
        supervised = os.environ.get("SPEAK_SUPERVISED") == "1"
        evict_interval = 3600  # check once per hour
        last_evict = time.monotonic()
        while True:
            await asyncio.sleep(30)
            if not supervised:
                idle_for = time.monotonic() - self.last_activity
                non_subscriber_conns = self.active_connections - self.subscriber_manager.count
                if non_subscriber_conns <= 0 and idle_for >= IDLE_TIMEOUT and not self.playback_queue.is_active:
                    print(f"speak-daemon: idle for {IDLE_TIMEOUT}s, shutting down", file=sys.stderr)
                    self._shutdown.set()
                    return
            if time.monotonic() - last_evict > evict_interval:
                removed = self.cache.evict_expired()
                if removed:
                    print(f"speak-daemon: evicted {removed} expired cache entries", file=sys.stderr)
                last_evict = time.monotonic()

    async def _startup_announce(self):
        """Announce startup, jumping to front of queue."""
        await self.playback_queue.enqueue({
            "text": "Speak daemon ready.",
            "enqueue": True,
            "voice": "af_heart",
            "speed": DEFAULT_SPEED,
            "lang": "en-us",
            "caller": "",
            "session": "",
        }, priority=True)

    def _request_shutdown(self, sig=None):
        """Signal the run loop to begin a graceful shutdown (idempotent)."""
        if sig is not None:
            print(
                f"speak-daemon: received {signal.Signals(sig).name}, shutting down",
                file=sys.stderr,
            )
        if self._shutdown is not None:
            self._shutdown.set()

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

        # Graceful shutdown coordination: SIGTERM/SIGINT and the unsupervised
        # idle watchdog all set this event. add_signal_handler keeps signal
        # delivery on the event loop (no sys.exit from a handler mid-coroutine),
        # and the finally block unlinks the socket/pid exactly once. Returning
        # normally keeps the process exit code at 0 for launchd bootout.
        self._shutdown = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._request_shutdown, sig)

        s = self.cache.stats()
        print(
            f"speak-daemon: listening on {SOCKET_PATH} (pid {os.getpid()})\n"
            f"  cache: {s['clauses']} clauses ({s['clause_hits']} hits), "
            f"{s['words']} words ({s['word_hits']} hits), TTL={CACHE_TTL_DAYS}d",
            file=sys.stderr,
        )

        self.playback_queue.start()
        asyncio.create_task(self.idle_watchdog())

        # Announce startup — schedule as task so the server is serving first.
        asyncio.create_task(self._startup_announce())

        try:
            async with server:
                await self._shutdown.wait()
        finally:
            # Graceful shutdown: close the history DB connection, then unlink the
            # socket/pid files. close_history() swallows its own errors so a bad
            # close never blocks socket cleanup.
            self.playback_queue.close_history()
            _cleanup_files()


# ---------------------------------------------------------------------------
# Command registry
#
# Each handler is `async def(daemon, request, writer) -> dict | _HANDLED`.
# Returning a dict sends it as the JSON reply and closes the connection.
# Returning _HANDLED means the handler already managed the reply / connection
# (used by `subscribe`, which holds the socket open for broadcast frames) and
# the dispatcher must not touch it further. Handlers reach queue state only
# through PlaybackQueue's public API — never its underscore internals.
# ---------------------------------------------------------------------------

_HANDLED = object()


async def _cmd_skip(daemon, request, writer):
    return await daemon.playback_queue.skip()


async def _cmd_clear(daemon, request, writer):
    return await daemon.playback_queue.clear()


async def _cmd_queue_status(daemon, request, writer):
    return daemon.playback_queue.status()


async def _cmd_pause(daemon, request, writer):
    return await daemon.playback_queue.pause()


async def _cmd_resume(daemon, request, writer):
    return daemon.playback_queue.resume()


async def _cmd_toggle_pause(daemon, request, writer):
    return await daemon.playback_queue.toggle_pause()


async def _cmd_replay(daemon, request, writer):
    return await daemon.playback_queue.replay()


async def _cmd_replay_by_id(daemon, request, writer):
    row_id = request.get("id")
    if row_id is None:
        return {"ok": False, "error": "replay_by_id requires 'id' field"}
    from_clause = request.get("from_clause")
    return await daemon.playback_queue.replay_by_id(int(row_id), from_clause=from_clause)


async def _cmd_stats(daemon, request, writer):
    q = daemon.playback_queue
    return {
        "daemon": {
            "pid": os.getpid(),
            "uptime_secs": round(time.time() - daemon.start_time),
            "active_connections": daemon.active_connections,
        },
        "queue": {
            "total_enqueued": q.total_enqueued,
            "total_completed": q.total_completed,
            "total_skipped": q.total_skipped,
            "pending": q.pending_count(),
            "playing": q.current_summary(),
            "resume_mid_clause": q.resume_mid_clause,
        },
        "cache": daemon.cache.stats(),
        "subscribers": daemon.subscriber_manager.status(),
    }


async def _cmd_voice_pool_status(daemon, request, writer):
    return {"ok": True, **daemon.voice_pool.status()}


async def _cmd_voice_release(daemon, request, writer):
    voice = request.get("voice", "")
    if not voice:
        return {"ok": False, "error": "voice_release requires 'voice' field"}
    released = daemon.voice_pool.release_voice(voice)
    return {"ok": True, "released": released}


async def _cmd_list_devices(daemon, request, writer):
    import sounddevice as sd
    devices = sd.query_devices()
    default_out = sd.default.device[1]
    output_devices = []
    for i, d in enumerate(devices):
        if d["max_output_channels"] > 0:
            output_devices.append({
                "index": i,
                "name": d["name"],
                "channels": d["max_output_channels"],
                "default": i == default_out,
            })
    return {"ok": True, "devices": output_devices}


async def _cmd_set_device(daemon, request, writer):
    device = request.get("device")
    if device is None:
        return {"ok": False, "error": "set_device requires 'device' field (int index or string name)"}
    # Validate the device before switching.
    import sounddevice as sd
    try:
        if isinstance(device, int):
            info = sd.query_devices(device)
            if info["max_output_channels"] == 0:
                return {"ok": False, "error": f"device {device} has no output channels"}
            await daemon.playback_queue.set_device(device)
            return {"ok": True, "device": {"index": device, "name": info["name"]}}
        # String name — resolve to an index by substring match.
        devices = sd.query_devices()
        needle = str(device).lower()
        matched = None
        for i, d in enumerate(devices):
            if d["max_output_channels"] > 0 and needle in d["name"].lower():
                matched = i
                break
        if matched is None:
            return {"ok": False, "error": f"no output device matching '{device}'"}
        await daemon.playback_queue.set_device(matched)
        info = sd.query_devices(matched)
        return {"ok": True, "device": {"index": matched, "name": info["name"]}}
    except Exception as e:
        return {"ok": False, "error": str(e)}


async def _cmd_history(daemon, request, writer):
    n = request.get("n", 10)
    offset = request.get("offset", 0)
    entries, total = daemon.playback_queue.history_get(n, offset)
    return {"ok": True, "entries": entries, "total": total}


async def _cmd_session_history(daemon, request, writer):
    session = request.get("session", "")
    n = request.get("n", 10)
    offset = request.get("offset", 0)
    entries, total = daemon.playback_queue.history_by_session(session, n, offset)
    return {"ok": True, "entries": entries, "total": total}


async def _cmd_caller_history(daemon, request, writer):
    caller = request.get("caller", "")
    n = request.get("n", 10)
    offset = request.get("offset", 0)
    entries, total = daemon.playback_queue.history_by_caller(caller, n, offset)
    return {"ok": True, "entries": entries, "total": total}


async def _cmd_caller_voice(daemon, request, writer):
    caller = request.get("caller", "")
    if not caller:
        return {"ok": False, "error": "caller_voice requires 'caller' field"}
    voice = daemon.playback_queue.last_voice_for_caller(caller)
    return {"ok": True, "caller": caller, "voice": voice}


async def _cmd_set_resume_mid_clause(daemon, request, writer):
    q = daemon.playback_queue
    enabled = request.get("enabled")
    if enabled is None:
        return {"ok": True, "resume_mid_clause": q.resume_mid_clause}
    q.resume_mid_clause = bool(enabled)
    return {"ok": True, "resume_mid_clause": q.resume_mid_clause}


async def _cmd_play_tone(daemon, request, writer):
    session = request.get("session", "")
    waveform = request.get("waveform", "pluck")
    if not session:
        return {"ok": False, "error": "play_tone requires 'session' field"}
    from .tones import get_input_tone, get_caller_tone
    tone_pcm = get_input_tone(session) if waveform == "pluck" else get_caller_tone(session)
    await daemon.playback_queue.play_raw_pcm(tone_pcm)
    return {"ok": True, "session": session, "waveform": waveform}


async def _cmd_synth(daemon, request, writer):
    """Synthesize text and return WAV bytes (base64) — no local playback.

    Serves remote surfaces (e.g. the Parlay panel) that play the audio on
    their own device. Reuses the same kokoro streaming path as playback
    rendering so pronunciation, clause splitting, and voice-pool behavior
    match spoken output. CPU inference runs on kokoro's internal thread via
    create_stream, so the event loop is not starved for long texts.
    """
    import base64
    import io
    import wave

    import numpy as np

    from .text import split_clauses

    text = str(request.get("text", "")).strip()[:2000]
    if not text:
        return {"ok": False, "error": "synth requires 'text'"}

    caller = str(request.get("caller", "") or "")
    session = str(request.get("session", "") or "")
    requested_voice = str(request.get("voice", "") or "")
    if requested_voice:
        voice_name = requested_voice
    elif caller:
        voice_name, _gain, _is_new = daemon.voice_pool.get_voice(caller, session, "af_heart")
    else:
        voice_name = "af_heart"
    speed = float(request.get("speed", DEFAULT_SPEED))
    lang = str(request.get("lang", "en-us"))

    daemon._touch_activity()
    chunks = []
    for clause in split_clauses(text):
        async for audio, _sr in daemon.synth.kokoro.create_stream(
            clause, voice_name, speed, lang, trim=False
        ):
            chunks.append(audio.squeeze())
    if not chunks:
        return {"ok": False, "error": "no audio produced"}

    full = np.concatenate(chunks)
    pcm = (full * 32767).astype(np.int16).tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(pcm)

    return {
        "ok": True,
        "voice": voice_name,
        "sample_rate": 24000,
        "seconds": round(len(full) / 24000, 2),
        "wav_b64": base64.b64encode(buf.getvalue()).decode(),
    }


async def _cmd_subscribe(daemon, request, writer):
    include_metadata = request.get("include_metadata", True)
    send_json(writer, {
        "ok": True, "subscribed": True,
        "sample_rate": 24000, "channels": 1, "format": "s16le",
    })
    await writer.drain()
    # Send current state so the subscriber has immediate context.
    current = daemon.playback_queue.current_broadcast()
    if current is not None:
        daemon.subscriber_manager.broadcast_metadata({
            "event": "item_start",
            "playing": current,
        })
    info = daemon.subscriber_manager.add(writer, include_metadata)
    # Keep the connection alive until the subscriber disconnects.
    await info.disconnect_event.wait()
    return _HANDLED


COMMANDS = {
    "skip": _cmd_skip,
    "clear": _cmd_clear,
    "queue_status": _cmd_queue_status,
    "pause": _cmd_pause,
    "resume": _cmd_resume,
    "toggle_pause": _cmd_toggle_pause,
    "replay": _cmd_replay,
    "replay_by_id": _cmd_replay_by_id,
    "stats": _cmd_stats,
    "voice_pool_status": _cmd_voice_pool_status,
    "voice_release": _cmd_voice_release,
    "list_devices": _cmd_list_devices,
    "set_device": _cmd_set_device,
    "history": _cmd_history,
    "session_history": _cmd_session_history,
    "caller_history": _cmd_caller_history,
    "caller_voice": _cmd_caller_voice,
    "set_resume_mid_clause": _cmd_set_resume_mid_clause,
    "play_tone": _cmd_play_tone,
    "synth": _cmd_synth,
    "subscribe": _cmd_subscribe,
}


def _cleanup_files():
    """Unlink the socket and pid file. Best-effort; safe to call once on exit."""
    for path in (SOCKET_PATH, SOCKET_PATH + ".pid"):
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


def main():
    import argparse

    # Apply Kokoro speed bug monkeypatch before anything else
    apply_patch()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--voices", required=True)
    parser.add_argument("--device", default=None, help="Audio output device (index or name substring)")
    args = parser.parse_args()

    # Parse device: try int first, else keep as string for name matching
    device = args.device
    if device is not None:
        try:
            device = int(device)
        except ValueError:
            # String name — resolve to index at startup
            import sounddevice as sd
            devices = sd.query_devices()
            needle = device.lower()
            matched = None
            for i, d in enumerate(devices):
                if d["max_output_channels"] > 0 and needle in d["name"].lower():
                    matched = i
                    break
            if matched is None:
                print(f"speak-daemon: no output device matching '{device}'", file=sys.stderr)
                sys.exit(1)
            print(f"speak-daemon: resolved device '{device}' -> index {matched} ({devices[matched]['name']})", file=sys.stderr)
            device = matched

    # Signal handling for clean shutdown is installed inside run() via
    # loop.add_signal_handler once the event loop exists. Before the loop
    # starts (during model load) there is no socket/pid to clean, so the
    # default SIGTERM disposition is safe.

    print("speak-daemon: loading model...", file=sys.stderr)
    daemon = SpeakDaemon(args.model, args.voices, device=device)
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
