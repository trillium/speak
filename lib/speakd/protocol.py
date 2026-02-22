"""Wire protocol helpers: length-prefixed JSON, state publishing, and event log."""

import json
import os
import struct
import time

from .config import EVENT_LOG_PATH, STATE_PATH


def send_json(writer, obj: dict) -> None:
    """Send a JSON response using the length-prefixed protocol, then zero terminator."""
    payload = json.dumps(obj).encode()
    writer.write(struct.pack("!I", len(payload)))
    writer.write(payload)
    writer.write(struct.pack("!I", 0))


def publish_state(state: dict) -> None:
    """Write current state to a JSON file for external tools to monitor."""
    state["timestamp"] = time.time()
    tmp = STATE_PATH + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(state, f)
        os.replace(tmp, STATE_PATH)
    except OSError:
        pass


def log_event(event: str, **data) -> None:
    """Append a structured JSONL event to the event log."""
    entry = {"ts": time.monotonic(), "wall": time.time(), "event": event}
    entry.update(data)
    try:
        with open(EVENT_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass
