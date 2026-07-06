"""Wire protocol helpers: length-prefixed JSON, state publishing, and event log."""

import json
import os
import struct
import time

from .config import EVENT_LOG_PATH, STATE_PATH

# Broadcast frame type tags
FRAME_TYPE_AUDIO = 0x01
FRAME_TYPE_METADATA = 0x02


def encode_broadcast_frame(frame_type: int, payload: bytes) -> bytes:
    """Encode a broadcast frame: [4-byte BE length][1-byte type][payload]."""
    total_len = 1 + len(payload)
    return struct.pack("!IB", total_len, frame_type) + payload


def send_json(writer, obj: dict) -> None:
    """Send a JSON response using the length-prefixed protocol, then zero terminator."""
    payload = json.dumps(obj).encode()
    writer.write(struct.pack("!I", len(payload)))
    writer.write(payload)
    writer.write(struct.pack("!I", 0))


# publish_state is called on every queue event with a full-state disk write.
# To kill write bursts (e.g. rapid consecutive "enqueued" events) we skip the
# disk write when it would land within _THROTTLE_SECS of the previous write AND
# carries the same event kind. Any change of event kind — the transitions that
# matter (enqueued/playing/item_done/idle) — always writes immediately. The
# subscriber broadcast in PlaybackQueue._publish is separate and stays
# unthrottled.
_THROTTLE_SECS = 0.1
_last_write_monotonic = 0.0
_last_event: str | None = None


def publish_state(state: dict) -> None:
    """Write current state to a JSON file for external tools to monitor.

    Throttled: bursts of same-kind events within _THROTTLE_SECS collapse to one
    write; a different event kind always writes.
    """
    global _last_write_monotonic, _last_event
    event = state.get("event")
    now = time.monotonic()
    if event == _last_event and (now - _last_write_monotonic) < _THROTTLE_SECS:
        return

    state["timestamp"] = time.time()
    tmp = STATE_PATH + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(state, f)
        os.replace(tmp, STATE_PATH)
    except OSError:
        return
    _last_write_monotonic = now
    _last_event = event


# Event log rotation: the JSONL log in /tmp otherwise grows unbounded. Every
# _EVENT_LOG_ROTATE_CHECK_EVERY calls we stat the file and, if it exceeds
# _EVENT_LOG_MAX_BYTES, rotate it to .1 before appending. The stat is throttled
# so the common path stays a single append with no extra syscall.
_EVENT_LOG_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
_EVENT_LOG_ROTATE_CHECK_EVERY = 1000
_event_log_calls = 0


def log_event(event: str, **data) -> None:
    """Append a structured JSONL event to the event log (rotates past 10 MB)."""
    global _event_log_calls
    _event_log_calls += 1
    if _event_log_calls % _EVENT_LOG_ROTATE_CHECK_EVERY == 0:
        try:
            if os.path.getsize(EVENT_LOG_PATH) > _EVENT_LOG_MAX_BYTES:
                os.replace(EVENT_LOG_PATH, EVENT_LOG_PATH + ".1")
        except OSError:
            pass

    entry = {"ts": time.monotonic(), "wall": time.time(), "event": event}
    entry.update(data)
    try:
        with open(EVENT_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass
