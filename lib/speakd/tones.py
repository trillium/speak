"""Separator tones, caller identification tones, and voice assignments.

Two waveform families share the same pitch patterns:
  - "sine" (default) — smooth, used for speech caller tones
  - "pluck" — sharp attack with decay, used for input-received confirmation

Same pitches = same identity. Different timbre = different event type.
"""

import hashlib

import numpy as np

from .config import SAMPLE_RATE


# Tone patterns — expanded set for fewer collisions across many sessions.
# Mix of 1-note, 2-note, and 3-note patterns across a wide pitch range.
TONE_SETS = [
    # 1-note: distinct pitches spanning two octaves
    [(261.63,)],   # C4
    [(293.66,)],   # D4
    [(329.63,)],   # E4
    [(392.00,)],   # G4
    [(440.00,)],   # A4
    [(523.25,)],   # C5
    [(587.33,)],   # D5
    [(659.25,)],   # E5
    [(783.99,)],   # G5
    [(880.00,)],   # A5
    # 2-note: ascending intervals
    [(261.63, 392.00)],   # C4 -> G4 (fifth)
    [(293.66, 440.00)],   # D4 -> A4 (fifth)
    [(329.63, 523.25)],   # E4 -> C5 (minor sixth)
    [(392.00, 587.33)],   # G4 -> D5 (fifth)
    [(440.00, 659.25)],   # A4 -> E5 (fifth)
    [(523.25, 783.99)],   # C5 -> G5 (fifth)
    # 2-note: descending intervals
    [(523.25, 329.63)],   # C5 -> E4 (descending)
    [(659.25, 440.00)],   # E5 -> A4 (descending)
    [(783.99, 523.25)],   # G5 -> C5 (descending)
    [(880.00, 587.33)],   # A5 -> D5 (descending)
    # 2-note: octave leaps
    [(261.63, 523.25)],   # C4 -> C5
    [(293.66, 587.33)],   # D4 -> D5
    [(329.63, 659.25)],   # E4 -> E5
    [(440.00, 880.00)],   # A4 -> A5
    # 3-note: arpeggios and melodic fragments
    [(261.63, 329.63, 392.00)],   # C4 -> E4 -> G4 (C major)
    [(293.66, 392.00, 523.25)],   # D4 -> G4 -> C5 (rising)
    [(329.63, 440.00, 587.33)],   # E4 -> A4 -> D5 (rising)
    [(392.00, 523.25, 659.25)],   # G4 -> C5 -> E5 (major arp)
    [(440.00, 523.25, 659.25)],   # A4 -> C5 -> E5 (Am arp)
    [(523.25, 659.25, 880.00)],   # C5 -> E5 -> A5 (wide rising)
    [(659.25, 523.25, 392.00)],   # E5 -> C5 -> G4 (descending)
    [(880.00, 659.25, 523.25)],   # A5 -> E5 -> C5 (descending)
    [(587.33, 440.00, 329.63)],   # D5 -> A4 -> E4 (descending)
    [(783.99, 587.33, 440.00)],   # G5 -> D5 -> A4 (descending)
    [(261.63, 440.00, 659.25)],   # C4 -> A4 -> E5 (wide skip)
    [(329.63, 523.25, 880.00)],   # E4 -> C5 -> A5 (wide skip)
]


def _freqs_for_key(key: str) -> tuple[float, ...]:
    """Deterministically select a pitch pattern from a string key."""
    h = int(hashlib.md5(key.encode()).hexdigest()[:8], 16)
    return TONE_SETS[h % len(TONE_SETS)][0]


def _generate_sine_note(freq: float, duration: float, volume: float) -> np.ndarray:
    """Pure sine wave with fade envelope — smooth, warm."""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), dtype=np.float32)
    note = np.sin(2 * np.pi * freq * t) * volume
    fade_len = int(SAMPLE_RATE * 0.015)
    note[:fade_len] *= np.linspace(0, 1, fade_len)
    note[-fade_len:] *= np.linspace(1, 0, fade_len)
    return note


def _generate_pluck_note(freq: float, duration: float, volume: float) -> np.ndarray:
    """Soft pluck — rounded attack, gentle decay. Same pitch as sine, warmer character."""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), dtype=np.float32)
    # Fundamental-heavy mix with soft upper harmonics
    wave = (
        np.sin(2 * np.pi * freq * t) * 0.75        # strong fundamental
        + np.sin(2 * np.pi * freq * 2 * t) * 0.15   # gentle octave
        + np.sin(2 * np.pi * freq * 3 * t) * 0.05   # hint of fifth
    )
    # Slower decay — sustains longer, less percussive
    decay = np.exp(-t * 6)  # τ ≈ 160ms (was 80ms)
    # Soft attack ramp — removes the initial click
    attack_len = int(SAMPLE_RATE * 0.008)  # 8ms fade-in
    attack = np.ones_like(t)
    attack[:attack_len] = np.linspace(0, 1, attack_len)
    note = wave * decay * attack * volume
    return note.astype(np.float32)


def _build_tone(freqs: tuple[float, ...], waveform: str = "sine") -> bytes:
    """Assemble a multi-note tone from a pitch pattern and waveform type."""
    gen = _generate_pluck_note if waveform == "pluck" else _generate_sine_note
    duration = {1: 0.16, 2: 0.12, 3: 0.08}[len(freqs)]
    volume = 0.10
    gap = np.zeros(int(SAMPLE_RATE * 0.04), dtype=np.float32)

    parts = [np.zeros(int(SAMPLE_RATE * 0.04), dtype=np.float32)]  # leading silence
    for i, freq in enumerate(freqs):
        parts.append(gen(freq, duration, volume))
        if i < len(freqs) - 1:
            parts.append(gap)
    parts.append(np.zeros(int(SAMPLE_RATE * 0.06), dtype=np.float32))  # trailing silence

    tone = np.concatenate(parts)
    pcm_int16 = (tone * 32767).astype(np.int16)
    return pcm_int16.tobytes()


# --- Separator tone (unchanged) ---

def _generate_separator_tone() -> bytes:
    """Gentle two-note chime to separate queue items. E5 -> G5, ~300ms."""
    duration = 0.15
    volume = 0.08
    t1 = np.linspace(0, duration, int(SAMPLE_RATE * duration), dtype=np.float32)
    t2 = np.linspace(0, duration, int(SAMPLE_RATE * duration), dtype=np.float32)
    note1 = np.sin(2 * np.pi * 659 * t1) * volume
    note2 = np.sin(2 * np.pi * 784 * t2) * volume
    fade_len = int(SAMPLE_RATE * 0.03)
    for note in (note1, note2):
        note[:fade_len] *= np.linspace(0, 1, fade_len)
        note[-fade_len:] *= np.linspace(1, 0, fade_len)
    silence_before = np.zeros(int(SAMPLE_RATE * 0.05), dtype=np.float32)
    gap = np.zeros(int(SAMPLE_RATE * 0.03), dtype=np.float32)
    silence_after = np.zeros(int(SAMPLE_RATE * 0.08), dtype=np.float32)
    tone = np.concatenate([silence_before, note1, gap, note2, silence_after])
    pcm_int16 = (tone * 32767).astype(np.int16)
    return pcm_int16.tobytes()


SEPARATOR_TONE = _generate_separator_tone()
CALLER_GAP = np.zeros(int(SAMPLE_RATE * 1.0), dtype=np.int16).tobytes()


# --- Caller tones (sine — for speech) ---

_caller_tone_cache: dict[str, bytes] = {}


def get_caller_tone(caller: str) -> bytes:
    """Sine wave caller tone — played before/after speech."""
    if caller not in _caller_tone_cache:
        freqs = _freqs_for_key(caller)
        _caller_tone_cache[caller] = _build_tone(freqs, waveform="sine")
    return _caller_tone_cache[caller]


# --- Input tones (pluck — for prompt received) ---

_input_tone_cache: dict[str, bytes] = {}


def get_input_tone(session: str) -> bytes:
    """Pluck waveform tone — same pitches as caller, different timbre.

    Keyed on session (PID) so each window gets its own identity.
    """
    if session not in _input_tone_cache:
        freqs = _freqs_for_key(session)
        _input_tone_cache[session] = _build_tone(freqs, waveform="pluck")
    return _input_tone_cache[session]

