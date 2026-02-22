"""Configuration constants for the speak daemon."""

import os
import pathlib

# --- Socket and state paths ---
SOCKET_PATH = f"/tmp/speak-{os.environ['USER']}.sock"
STATE_PATH = f"/tmp/speak-{os.environ['USER']}.state.json"

# --- Timeouts ---
IDLE_TIMEOUT = 300  # shut down after 5 minutes idle

# --- Cache ---
CACHE_DIR = pathlib.Path(os.environ.get(
    "SPEAK_CACHE_DIR",
    f"/tmp/speak-cache-{os.environ['USER']}",
))
CACHE_TTL_DAYS = int(os.environ.get("SPEAK_CACHE_TTL_DAYS", "3"))

# --- Audio ---
SAMPLE_RATE = 24000
CROSSFADE_MS = 5          # crossfade ramp at word joins (avoids clicks)
SILENCE_GAP_MS = 30       # silence inserted between assembled words
CROSSFADE_SAMPLES = int(SAMPLE_RATE * CROSSFADE_MS / 1000)
SILENCE_SAMPLES = int(SAMPLE_RATE * SILENCE_GAP_MS / 1000)

# Energy threshold for silence detection (relative to peak)
SILENCE_THRESHOLD = 0.02
SILENCE_MIN_SAMPLES = int(SAMPLE_RATE * 0.02)  # 20ms minimum gap to count as word boundary
