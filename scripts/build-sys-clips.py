#!/usr/bin/env python3
"""Build pre-synthesized system voice clips for blob label replacement.

Defaults to am_adam at speed=0.85 — the dedicated sys voice (override with
--voice / --speed). Output: raw int16 PCM at the daemon sample rate
(speakd.config.SAMPLE_RATE), no WAV header.
Output dir: ~/.local/share/speak/sys-clips/

Usage:
    python3 build-sys-clips.py                     # build all clips
    python3 build-sys-clips.py --label "hash"      # build one
    python3 build-sys-clips.py --voice am_onyx --speed 0.9
"""
import argparse
import os
import pathlib
import sys

import numpy as np

# Point at the daemon lib
_LIB = pathlib.Path(__file__).parent.parent / "lib"
sys.path.insert(0, str(_LIB))

from kokoro_onnx import Kokoro

from speakd.config import SAMPLE_RATE  # single source of truth for the 24kHz rate

# am_adam @ 0.85 is the dedicated sys voice; overridable via --voice / --speed.
DEFAULT_VOICE = "am_adam"
DEFAULT_SPEED = 0.85
LANG = "en-us"
SILENCE_THRESH = 0.001
PAD_MS = 50

MODEL = pathlib.Path.home() / ".local/share/speak/kokoro/kokoro-v1.0.onnx"
VOICES = pathlib.Path.home() / ".local/share/speak/kokoro/voices-v1.0.bin"
OUT_DIR = pathlib.Path.home() / ".local/share/speak/sys-clips"

# Complete label -> slug mapping (spaces->underscores, hyphens->underscores)
LABELS = [
    # Short SHA / generic hash
    "hash",
    # SHA-1 full
    "commit hash",
    # UUID
    "UUID",
    # MD5
    "MD5 hash",
    # SHA-256
    "SHA-256 hash",
    # SHA-512
    "SHA-512 hash",
    # Generic hex
    "hex string",
    # Base64
    "base64 data",
    "base64 image",
    # JWT
    "JWT token",
    # API keys (generic + specific)
    "API key",
    "Stripe secret key",
    "Stripe publishable key",
    "GitHub token",
    "Slack bot token",
    "Slack app token",
    "Google API key",
    "Hugging Face token",
    "AWS access key",
    # PEM blocks
    "private key block",
    "certificate block",
    "public key block",
    "certificate request",
    "key block",
    # Network
    "IP address",
    "IPv6 address",
    "MAC address",
    # Stripe-prefixed IDs
    "payment intent ID",
    "charge ID",
    "subscription ID",
    "customer ID",
    "product ID",
    "price ID",
    "invoice ID",
    "refund ID",
    "event ID",
    "request ID",
    "token ID",
    "card ID",
    "bank account ID",
    "account ID",
    "service ID",
    # Other
    "hex dump",
    "snowflake ID",
    "large numeric ID",
    "large ID",
]


def label_to_slug(label: str) -> str:
    return label.replace(" ", "_").replace("-", "_")


def find_voice_bounds(audio: np.ndarray) -> tuple[int, int]:
    abs_audio = np.abs(audio)
    peak = np.max(abs_audio)
    if peak == 0:
        return 0, len(audio)
    threshold = peak * SILENCE_THRESH
    above = np.where(abs_audio > threshold)[0]
    if len(above) == 0:
        return 0, len(audio)
    return int(above[0]), int(above[-1]) + 1


def build_clip(kokoro: Kokoro, label: str, voice_style, speed: float) -> bytes:
    phonemes = kokoro.tokenizer.phonemize(label, LANG)
    audio, sr = kokoro._create_audio(phonemes, voice_style, speed)
    audio = np.squeeze(audio)

    start, end = find_voice_bounds(audio)
    trimmed = audio[start:end]

    pad_samples = int(SAMPLE_RATE * PAD_MS / 1000)
    pad = np.zeros(pad_samples, dtype=np.float32)
    padded = np.concatenate([pad, trimmed, pad])

    pcm = (np.clip(padded, -1.0, 1.0) * 32767).astype(np.int16)
    return pcm.tobytes()


def main():
    parser = argparse.ArgumentParser(description="Build speak sys clips")
    parser.add_argument("--label", help="Build only this label (exact match)")
    parser.add_argument("--voice", default=DEFAULT_VOICE,
                        help=f"Kokoro voice (default: {DEFAULT_VOICE})")
    parser.add_argument("--speed", type=float, default=DEFAULT_SPEED,
                        help=f"Speech speed (default: {DEFAULT_SPEED})")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading Kokoro model...")
    kokoro = Kokoro(str(MODEL), str(VOICES))
    voice_style = kokoro.get_voice_style(args.voice)
    print(f"Model loaded. Voice: {args.voice} speed={args.speed}")

    labels = [args.label] if args.label else LABELS

    for label in labels:
        slug = label_to_slug(label)
        out_path = OUT_DIR / f"{slug}.pcm"
        try:
            pcm = build_clip(kokoro, label, voice_style, args.speed)
            out_path.write_bytes(pcm)
            dur_ms = len(pcm) // 2 / SAMPLE_RATE * 1000
            print(f"  {slug}.pcm  ({dur_ms:.0f}ms, {len(pcm)} bytes)")
        except Exception as e:
            print(f"  FAIL {slug}: {e}")

    print(f"\nDone. Clips at: {OUT_DIR}")


if __name__ == "__main__":
    main()
