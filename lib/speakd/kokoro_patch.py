"""Monkeypatch for kokoro-onnx speed dtype bug.

Fix kokoro-onnx bug: speed dtype is np.int32 instead of np.float32
for models using input_ids format (v1.0 models).
The bug: np.array([1.8], dtype=np.int32) truncates to [1], losing the
fractional speed value. We must inject the original float BEFORE int truncation.
"""

import numpy as np
from kokoro_onnx import Kokoro

_orig_create_audio = Kokoro._create_audio


def _fixed_create_audio(self, phonemes, voice, speed):
    orig_run = self.sess.run
    def patched_run(output_names, inputs, *args, **kwargs):
        if "speed" in inputs:
            inputs["speed"] = np.array([speed], dtype=np.float32)
        return orig_run(output_names, inputs, *args, **kwargs)
    self.sess.run = patched_run
    try:
        return _orig_create_audio(self, phonemes, voice, speed)
    finally:
        self.sess.run = orig_run


def apply_patch():
    """Apply the speed dtype monkeypatch to Kokoro."""
    Kokoro._create_audio = _fixed_create_audio
