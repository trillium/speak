"""Audio processing: word boundary detection and word audio assembly."""

import numpy as np

from .config import (
    CROSSFADE_SAMPLES,
    SAMPLE_RATE,
    SILENCE_MIN_SAMPLES,
    SILENCE_SAMPLES,
    SILENCE_THRESHOLD,
)


def detect_word_boundaries(audio: np.ndarray, n_words: int) -> list[tuple[int, int]]:
    """Find word boundaries in synthesized audio using energy-based silence detection.

    Returns list of (start, end) sample indices for each word segment.
    Falls back to equal division if silence detection doesn't find enough boundaries.
    """
    # Compute short-time energy (5ms frames)
    frame_len = int(SAMPLE_RATE * 0.005)
    n_frames = len(audio) // frame_len
    if n_frames == 0:
        return [(0, len(audio))]

    frames = audio[:n_frames * frame_len].reshape(n_frames, frame_len)
    energy = np.mean(frames ** 2, axis=1)
    peak_energy = np.max(energy) if np.max(energy) > 0 else 1.0
    is_silent = energy < (peak_energy * SILENCE_THRESHOLD)

    # Find silent regions (consecutive silent frames)
    boundaries = []
    in_silence = False
    silence_start = 0
    for i, silent in enumerate(is_silent):
        if silent and not in_silence:
            silence_start = i
            in_silence = True
        elif not silent and in_silence:
            silence_len = (i - silence_start) * frame_len
            if silence_len >= SILENCE_MIN_SAMPLES:
                # Midpoint of the silence as the boundary
                mid_sample = ((silence_start + i) // 2) * frame_len
                boundaries.append(mid_sample)
            in_silence = False

    # We need exactly n_words - 1 boundaries
    if len(boundaries) < n_words - 1:
        # Not enough silence gaps found — fall back to equal division
        segment_len = len(audio) // n_words
        return [(i * segment_len, (i + 1) * segment_len) for i in range(n_words)]

    # Take the first n_words-1 boundaries since they're in order
    boundaries = boundaries[:n_words - 1]
    segments = []
    prev = 0
    for b in boundaries:
        segments.append((prev, b))
        prev = b
    segments.append((prev, len(audio)))
    return segments


def assemble_word_audio(word_pcm_list: list[bytes]) -> bytes:
    """Join word PCM segments with silence gaps and crossfade to avoid clicks."""
    if len(word_pcm_list) == 1:
        return word_pcm_list[0]

    silence = np.zeros(SILENCE_SAMPLES, dtype=np.int16)
    fade_in = np.linspace(0, 1, CROSSFADE_SAMPLES, dtype=np.float32)
    fade_out = np.linspace(1, 0, CROSSFADE_SAMPLES, dtype=np.float32)

    parts = []
    for i, pcm_bytes in enumerate(word_pcm_list):
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).copy()
        # Apply fade-out to end of each word (except last)
        if len(samples) > CROSSFADE_SAMPLES:
            if i < len(word_pcm_list) - 1:
                samples[-CROSSFADE_SAMPLES:] = (
                    samples[-CROSSFADE_SAMPLES:].astype(np.float32) * fade_out
                ).astype(np.int16)
            # Apply fade-in to start of each word (except first)
            if i > 0:
                samples[:CROSSFADE_SAMPLES] = (
                    samples[:CROSSFADE_SAMPLES].astype(np.float32) * fade_in
                ).astype(np.int16)
        parts.append(samples)
        if i < len(word_pcm_list) - 1:
            parts.append(silence)

    return np.concatenate(parts).tobytes()
