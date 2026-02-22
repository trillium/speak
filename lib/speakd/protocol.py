"""Wire protocol helpers: length-prefixed JSON and state publishing."""

import json
import os
import struct
import time

from .config import STATE_PATH


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
