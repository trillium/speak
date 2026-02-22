"""Separator tones, caller identification tones, and voice assignments."""

import hashlib

import numpy as np

from .config import SAMPLE_RATE


def _generate_separator_tone() -> bytes:
    """Generate a gentle two-note chime to separate queue items.

    A soft ascending two-tone (E5 -> G5) with fade in/out, ~300ms total.
    """
    duration = 0.15  # per note
    volume = 0.08    # gentle
    t1 = np.linspace(0, duration, int(SAMPLE_RATE * duration), dtype=np.float32)
    t2 = np.linspace(0, duration, int(SAMPLE_RATE * duration), dtype=np.float32)

    # E5 (659 Hz) then G5 (784 Hz)
    note1 = np.sin(2 * np.pi * 659 * t1) * volume
    note2 = np.sin(2 * np.pi * 784 * t2) * volume

    # Apply fade in/out envelope to each note
    fade_len = int(SAMPLE_RATE * 0.03)  # 30ms fade
    for note in (note1, note2):
        note[:fade_len] *= np.linspace(0, 1, fade_len)
        note[-fade_len:] *= np.linspace(1, 0, fade_len)

    # 50ms silence before, 30ms gap between notes, 80ms silence after
    silence_before = np.zeros(int(SAMPLE_RATE * 0.05), dtype=np.float32)
    gap = np.zeros(int(SAMPLE_RATE * 0.03), dtype=np.float32)
    silence_after = np.zeros(int(SAMPLE_RATE * 0.08), dtype=np.float32)

    tone = np.concatenate([silence_before, note1, gap, note2, silence_after])
    pcm_int16 = (tone * 32767).astype(np.int16)
    return pcm_int16.tobytes()


SEPARATOR_TONE = _generate_separator_tone()

# Silence gap between different callers (after end tone, before next start tone)
CALLER_GAP = np.zeros(int(SAMPLE_RATE * 1.0), dtype=np.int16).tobytes()  # 1s silence


def _generate_caller_tone(caller: str) -> bytes:
    """Generate a unique tone for a caller, derived from their name.

    Each caller gets:
    - A distinct number of beeps (1, 2, or 3) for easy aural identification
    - Unique pitches from the pentatonic scale
    - Beep count cycles through [1, 2, 3] based on caller ordering, so
      callers encountered in sequence always sound distinct
    """
    # Pentatonic scale frequencies — wide spread for maximum distinctness
    TONE_SETS = [
        # 1-beep patterns: single distinct pitch
        [(523.25,)],   # C5
        [(440.00,)],   # A4
        [(659.25,)],   # E5
        # 2-beep patterns: ascending or descending intervals
        [(329.63, 523.25)],   # E4 -> C5 (ascending 4th)
        [(783.99, 440.00)],   # G5 -> A4 (descending)
        [(293.66, 587.33)],   # D4 -> D5 (octave leap)
        # 3-beep patterns: melodic fragments
        [(392.00, 523.25, 659.25)],   # G4 -> C5 -> E5 (major arpeggio)
        [(880.00, 659.25, 523.25)],   # A5 -> E5 -> C5 (descending)
        [(329.63, 440.00, 587.33)],   # E4 -> A4 -> D5 (rising)
    ]
    h = int(hashlib.md5(caller.encode()).hexdigest()[:8], 16)
    pattern = TONE_SETS[h % len(TONE_SETS)]
    freqs = pattern[0]

    # Vary duration by beep count: 1-beep=longer, 3-beep=shorter
    duration = {1: 0.16, 2: 0.12, 3: 0.08}[len(freqs)]
    volume = 0.10
    fade_len = int(SAMPLE_RATE * 0.015)
    gap = np.zeros(int(SAMPLE_RATE * 0.04), dtype=np.float32)

    parts = [np.zeros(int(SAMPLE_RATE * 0.04), dtype=np.float32)]  # leading silence
    for i, freq in enumerate(freqs):
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration), dtype=np.float32)
        note = np.sin(2 * np.pi * freq * t) * volume
        note[:fade_len] *= np.linspace(0, 1, fade_len)
        note[-fade_len:] *= np.linspace(1, 0, fade_len)
        parts.append(note)
        if i < len(freqs) - 1:
            parts.append(gap)
    parts.append(np.zeros(int(SAMPLE_RATE * 0.06), dtype=np.float32))  # trailing silence

    tone = np.concatenate(parts)
    pcm_int16 = (tone * 32767).astype(np.int16)
    return pcm_int16.tobytes()


# Cache caller tones so we don't regenerate each time
_caller_tone_cache: dict[str, bytes] = {}


def get_caller_tone(caller: str) -> bytes:
    if caller not in _caller_tone_cache:
        _caller_tone_cache[caller] = _generate_caller_tone(caller)
    return _caller_tone_cache[caller]


# Voice assignments per caller. Callers not listed use the request's voice.
# This can be overridden via a config file in the future.
# Map caller names to (voice, gain) tuples.
# Gain 1.0 = no change, >1.0 = louder. Adjust per voice to normalize volume.
CALLER_VOICES: dict[str, tuple[str, float]] = {
    "speak": ("af_heart", 1.0),
    "happy": ("am_adam", 1.0),
    "ops":   ("af_nova", 1.5),     # nova is quiet by default
}


def get_caller_voice(caller: str, default_voice: str) -> tuple[str, float]:
    """Return (voice_name, gain) for a caller."""
    return CALLER_VOICES.get(caller, (default_voice, 1.0))
