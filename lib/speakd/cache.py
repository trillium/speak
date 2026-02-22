"""Two-tier audio cache: clause-level (high quality) and word-level (fast assembly).

Each entry consists of two files:
  <hash>      — raw PCM bytes (int16, 24kHz mono)
  <hash>.meta — JSON metadata: voice, speed, hits, created_at, label
"""

import hashlib
import json
import pathlib
import time

import numpy as np

from .audio import assemble_word_audio, detect_word_boundaries


class AudioCache:
    """Two-tier disk cache: clause-level (high quality) and word-level (fast assembly)."""

    def __init__(self, cache_dir: pathlib.Path, ttl_days: int):
        self.clause_dir = cache_dir / "clauses"
        self.word_dir = cache_dir / "words"
        self.ttl_secs = ttl_days * 86400
        self.clause_dir.mkdir(parents=True, exist_ok=True)
        self.word_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _hash(raw: str) -> str:
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    # --- metadata helpers ---

    @staticmethod
    def _read_meta(path: pathlib.Path) -> dict:
        try:
            return json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    @staticmethod
    def _write_meta(path: pathlib.Path, meta: dict) -> None:
        path.write_text(json.dumps(meta, separators=(",", ":")))

    @staticmethod
    def _bump_hits(meta_path: pathlib.Path) -> None:
        try:
            meta = json.loads(meta_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return
        meta["hits"] = meta.get("hits", 0) + 1
        meta["last_hit"] = time.time()
        meta_path.write_text(json.dumps(meta, separators=(",", ":")))

    # --- Clause cache (tier 1) ---

    def get_clause(self, text: str, voice: str, speed: float) -> bytes | None:
        h = self._hash(f"{text}|{voice}|{speed:.2f}")
        pcm_path = self.clause_dir / h
        meta_path = self.clause_dir / f"{h}.meta"
        if not pcm_path.exists():
            return None
        if time.time() - pcm_path.stat().st_mtime > self.ttl_secs:
            pcm_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
            return None
        self._bump_hits(meta_path)
        return pcm_path.read_bytes()

    def put_clause(self, text: str, voice: str, speed: float, pcm: bytes) -> None:
        h = self._hash(f"{text}|{voice}|{speed:.2f}")
        (self.clause_dir / h).write_bytes(pcm)
        meta_path = self.clause_dir / f"{h}.meta"
        existing = self._read_meta(meta_path)
        self._write_meta(meta_path, {
            "voice": voice, "speed": speed,
            "hits": existing.get("hits", 0),
            "created_at": existing.get("created_at", time.time()),
            "text": text[:200],
        })

    # --- Word cache (tier 2) ---

    def get_word(self, phonemes: str, voice: str, speed: float) -> bytes | None:
        h = self._hash(f"{phonemes}|{voice}|{speed:.2f}")
        pcm_path = self.word_dir / h
        meta_path = self.word_dir / f"{h}.meta"
        if not pcm_path.exists():
            return None
        if time.time() - pcm_path.stat().st_mtime > self.ttl_secs:
            pcm_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
            return None
        self._bump_hits(meta_path)
        return pcm_path.read_bytes()

    def put_word(self, phonemes: str, voice: str, speed: float, pcm: bytes) -> None:
        h = self._hash(f"{phonemes}|{voice}|{speed:.2f}")
        (self.word_dir / h).write_bytes(pcm)
        meta_path = self.word_dir / f"{h}.meta"
        existing = self._read_meta(meta_path)
        self._write_meta(meta_path, {
            "voice": voice, "speed": speed,
            "hits": existing.get("hits", 0),
            "created_at": existing.get("created_at", time.time()),
            "phonemes": phonemes,
        })

    def assemble_from_words(
        self, word_phonemes: list[str], voice: str, speed: float
    ) -> bytes | None:
        """Try to assemble full audio from cached words. Returns None if any word is missing."""
        word_pcms = []
        for wp in word_phonemes:
            pcm = self.get_word(wp, voice, speed)
            if pcm is None:
                return None
            word_pcms.append(pcm)
        return assemble_word_audio(word_pcms)

    def extract_and_cache_words(
        self, word_phonemes: list[str], audio: np.ndarray, voice: str, speed: float
    ) -> None:
        """Extract individual word audio from a full synthesis and cache each word."""
        if len(word_phonemes) <= 1:
            # Single word — cache the whole thing
            if word_phonemes:
                pcm = (audio * 32767).astype(np.int16).tobytes()
                self.put_word(word_phonemes[0], voice, speed, pcm)
            return

        segments = detect_word_boundaries(audio, len(word_phonemes))
        for wp, (start, end) in zip(word_phonemes, segments):
            segment = audio[start:end]
            if len(segment) > 0:
                pcm = (segment * 32767).astype(np.int16).tobytes()
                self.put_word(wp, voice, speed, pcm)

    def evict_expired(self) -> int:
        """Remove all expired entries across both tiers. Returns count removed."""
        now = time.time()
        removed = 0
        for d in (self.clause_dir, self.word_dir):
            for path in d.iterdir():
                if path.suffix == ".meta":
                    continue  # cleaned up with its PCM file
                if path.is_file() and (now - path.stat().st_mtime) > self.ttl_secs:
                    path.unlink(missing_ok=True)
                    pathlib.Path(f"{path}.meta").unlink(missing_ok=True)
                    removed += 1
        return removed

    def disk_size(self) -> int:
        """Return total disk usage of the cache in bytes."""
        total = 0
        for d in (self.clause_dir, self.word_dir):
            for path in d.iterdir():
                if path.is_file():
                    total += path.stat().st_size
        return total

    def stats(self) -> dict:
        """Return cache statistics including hit counts per voice."""
        result = {"clauses": 0, "words": 0, "clause_hits": 0, "word_hits": 0, "voices": {}}
        for tier, d in [("clause", self.clause_dir), ("word", self.word_dir)]:
            for meta_path in d.glob("*.meta"):
                meta = self._read_meta(meta_path)
                hits = meta.get("hits", 0)
                voice = meta.get("voice", "unknown")
                result[f"{tier}s"] += 1
                result[f"{tier}_hits"] += hits
                if voice not in result["voices"]:
                    result["voices"][voice] = {"clauses": 0, "words": 0, "hits": 0}
                result["voices"][voice][f"{tier}s"] += 1
                result["voices"][voice]["hits"] += hits
        result["disk_bytes"] = self.disk_size()
        return result
