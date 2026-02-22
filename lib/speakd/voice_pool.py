"""Voice pool: auto-assign distinct voices to caller sessions."""

import json
import os

ENGLISH_VOICES = [
    "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore",
    "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
    "am_michael", "am_onyx", "am_puck",
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
]


class VoicePool:
    def __init__(self, config_path: str):
        self._config_path = config_path
        self._locks, self._weights = self._load_config()
        self._claims: dict[tuple[str, str], tuple[str, float]] = {}
        self._next_idx = 0

    def _load_config(self) -> tuple[dict[str, tuple[str, float]], dict[str, int]]:
        try:
            with open(self._config_path) as f:
                data = json.load(f)
            locks = {
                name: (entry["voice"], entry.get("gain", 1.0))
                for name, entry in data.get("locks", {}).items()
            }
            weights = {
                voice: int(w)
                for voice, w in data.get("weights", {}).items()
            }
            return locks, weights
        except (FileNotFoundError, json.JSONDecodeError):
            return {}, {}

    def _save_config(self):
        data = {
            "locks": {
                name: {"voice": voice, "gain": gain}
                for name, (voice, gain) in self._locks.items()
            },
            "weights": self._weights,
        }
        os.makedirs(os.path.dirname(self._config_path), exist_ok=True)
        with open(self._config_path, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")

    def get_voice(self, caller: str, session: str, default_voice: str) -> tuple[str, float, bool]:
        """Return (voice, gain, is_new_claim) for a caller+session pair."""
        key = (caller, session)
        if key in self._claims:
            return *self._claims[key], False

        # Locked callers always get their locked voice
        if caller in self._locks:
            v, g = self._locks[caller]
            self._claims[key] = (v, g)
            return v, g, False

        # Pull from pool (excluding locked + claimed voices)
        locked_voices = {v for v, _ in self._locks.values()}
        claimed_voices = {v for v, _ in self._claims.values()}
        excluded = locked_voices | claimed_voices
        available = [v for v in ENGLISH_VOICES if v not in excluded]
        if not available:
            # All claimed — recycle, only excluding locked voices
            available = [v for v in ENGLISH_VOICES if v not in locked_voices]
        # Sort by weight ascending — low-weight voices assigned first
        available.sort(key=lambda v: self._weights.get(v, 0))
        voice = available[self._next_idx % len(available)]
        self._next_idx += 1
        self._claims[key] = (voice, 1.0)
        return voice, 1.0, True

    def lock(self, caller: str, voice: str, gain: float = 1.0):
        self._locks[caller] = (voice, gain)
        self._save_config()

    def unlock(self, caller: str) -> bool:
        if caller in self._locks:
            del self._locks[caller]
            self._save_config()
            return True
        return False

    def list_locks(self) -> dict:
        return {
            name: {"voice": voice, "gain": gain}
            for name, (voice, gain) in self._locks.items()
        }

    def set_weight(self, voice: str, weight: int):
        self._weights[voice] = weight
        self._save_config()

    def clear_weight(self, voice: str) -> bool:
        if voice in self._weights:
            del self._weights[voice]
            self._save_config()
            return True
        return False

    def list_weights(self) -> dict[str, int]:
        return dict(self._weights)

    def release_voice(self, voice: str) -> list[str]:
        """Remove all claims using the given voice. Returns released session keys."""
        released = []
        to_remove = [
            key for key, (v, _) in self._claims.items() if v == voice
        ]
        for key in to_remove:
            del self._claims[key]
            released.append(f"{key[0]}:{key[1]}")
        return released

    def status(self) -> dict:
        locks = self.list_locks()
        claims = {
            f"{caller}:{session}": {"voice": voice, "gain": gain}
            for (caller, session), (voice, gain) in self._claims.items()
        }
        return {"locks": locks, "claims": claims, "weights": self._weights}
