"""Configuration constants for the speak daemon."""

import json
import os
import pathlib

# --- Socket and state paths ---
SOCKET_PATH = f"/tmp/speak-{os.environ['USER']}.sock"
STATE_PATH = f"/tmp/speak-{os.environ['USER']}.state.json"

# --- Persistent settings (survive daemon restarts; deliberately NOT /tmp) ---
# Home for durable, user-chosen preferences the daemon honors on startup, e.g.
# a pinned audio output device. Kept separate from the volatile STATE_PATH so a
# launchd restart never loses the choice.
SETTINGS_DIR = pathlib.Path(os.environ.get(
    "SPEAK_SETTINGS_DIR",
    os.path.join(
        os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share")),
        "speak",
    ),
))
SETTINGS_PATH = SETTINGS_DIR / "settings.json"


def load_settings() -> dict:
    """Load persisted settings. Tolerant: missing or corrupt file yields {}."""
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, ValueError, OSError):
        return {}


def save_setting(key: str, value) -> bool:
    """Persist a single setting; value=None removes the key.

    Atomic write (tmp + os.replace). Best-effort: returns True on success,
    False on any filesystem error, and never raises so a set-device call can't
    crash the daemon over a disk hiccup.
    """
    settings = load_settings()
    if value is None:
        settings.pop(key, None)
    else:
        settings[key] = value
    try:
        SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
        tmp = SETTINGS_PATH.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
        os.replace(tmp, SETTINGS_PATH)
        return True
    except OSError:
        return False

# --- Timeouts ---
IDLE_TIMEOUT = 300  # shut down after 5 minutes idle

# --- Cache ---
CACHE_DIR = pathlib.Path(os.environ.get(
    "SPEAK_CACHE_DIR",
    f"/tmp/speak-cache-{os.environ['USER']}",
))
CACHE_TTL_DAYS = int(os.environ.get("SPEAK_CACHE_TTL_DAYS", "3"))

# --- Event log ---
EVENT_LOG_PATH = f"/tmp/speak-{os.environ['USER']}.events.jsonl"

# --- Audio ---
SAMPLE_RATE = 24000
DEFAULT_SPEED = 1.26  # default speech speed; bin/speak mirrors this
CROSSFADE_MS = 5          # crossfade ramp at word joins (avoids clicks)
SILENCE_GAP_MS = 30       # silence inserted between assembled words
CROSSFADE_SAMPLES = int(SAMPLE_RATE * CROSSFADE_MS / 1000)
SILENCE_SAMPLES = int(SAMPLE_RATE * SILENCE_GAP_MS / 1000)

# Energy threshold for word-boundary detection during word-audio assembly,
# as a fraction of peak frame energy (see audio.py detect_word_boundaries:
# `energy < peak_energy * SILENCE_THRESHOLD`). Deliberately coarse (0.02):
# it splits assembled words at inter-word gaps, so it must ignore the
# low-amplitude tails of voiced speech and only fire on real silence.
# Distinct from renderer._SILENCE_THRESH (0.001), which trims clause edges and
# is far more sensitive on purpose — see the note there.
SILENCE_THRESHOLD = 0.02
SILENCE_MIN_SAMPLES = int(SAMPLE_RATE * 0.02)  # 20ms minimum gap to count as word boundary
