"""TTS synthesis engine with two-tier cache integration."""

import numpy as np
from kokoro_onnx import Kokoro

from .cache import AudioCache


class SynthesisEngine:
    """Wraps Kokoro model with phonemization and two-tier cache logic."""

    def __init__(self, kokoro: Kokoro, cache: AudioCache):
        self.kokoro = kokoro
        self.cache = cache

    def phonemize_words(self, text: str, lang: str = "en-us") -> list[str]:
        """Phonemize each word of text independently. Returns list of phoneme strings."""
        words = text.split()
        return [self.kokoro.tokenizer.phonemize(w, lang) for w in words]

    def synthesize_full(self, text: str, voice_name: str, voice, speed: float, lang: str = "en-us") -> bytes:
        """Full synthesis: returns PCM bytes and populates both clause and word caches."""
        phonemes = self.kokoro.tokenizer.phonemize(text, lang)
        audio, sr = self.kokoro._create_audio(phonemes, voice, speed)
        pcm = (audio * 32767).astype(np.int16).tobytes()

        # Populate clause cache
        self.cache.put_clause(text, voice_name, speed, pcm)

        # Populate word cache by extracting individual words from the full audio
        word_phonemes = self.phonemize_words(text, lang)
        if len(word_phonemes) > 1:
            self.cache.extract_and_cache_words(word_phonemes, audio, voice_name, speed)
        elif word_phonemes:
            self.cache.put_word(word_phonemes[0], voice_name, speed, pcm)

        return pcm

    def synthesize_sentence(self, sentence: str, voice_name: str, voice, speed: float, lang: str = "en-us") -> tuple[bytes, bool]:
        """Synthesize a clause using two-tier cache: clause -> word assembly -> full synthesis.

        Returns (pcm_bytes, needs_background_upgrade) tuple.
        needs_background_upgrade is True when we served from word cache and want
        to do a full synthesis in the background for higher quality next time.
        """
        # Tier 1: clause cache (exact match, best quality)
        cached = self.cache.get_clause(sentence, voice_name, speed)
        if cached is not None:
            return cached, False

        # Tier 2: try to assemble from cached words
        word_phonemes = self.phonemize_words(sentence, lang)
        assembled = self.cache.assemble_from_words(word_phonemes, voice_name, speed)
        if assembled is not None:
            return assembled, True  # serve now, upgrade in background

        # Cache miss on both tiers: full synthesis
        pcm = self.synthesize_full(sentence, voice_name, voice, speed, lang)
        return pcm, False

    def bg_upgrade(self, sentence: str, voice_name: str, voice, speed: float, lang: str = "en-us") -> None:
        """Background task: do full synthesis to upgrade word-assembled cache to clause cache."""
        self.synthesize_full(sentence, voice_name, voice, speed, lang)
