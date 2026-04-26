"""Patch f5_tts_mlx for newer MLX; cache weights; play via WAV (afplay) to avoid PortAudio issues."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import einx
import mlx.core as mx

import f5_tts_mlx.utils as _f5_utils


def _lens_to_mask_compat(t, length=None):
    if length is None:
        length = int(mx.max(t).item())
    seq = mx.arange(length)
    return einx.less("n, b -> b n", seq, t)


_generate = None
_pretrained_cache_patched = False


def _patch_f5_pretrained_cache() -> None:
    global _pretrained_cache_patched
    if _pretrained_cache_patched:
        return
    from f5_tts_mlx.cfm import F5TTS

    orig_fn = F5TTS.__dict__["from_pretrained"].__func__
    cache: dict = {}

    def wrapped(cls, hf_model_name_or_path, convert_weights=False, bit=None):
        key = (hf_model_name_or_path, convert_weights, bit)
        if key not in cache:
            cache[key] = orig_fn(
                cls,
                hf_model_name_or_path,
                convert_weights=convert_weights,
                bit=bit,
            )
        return cache[key]

    F5TTS.from_pretrained = classmethod(wrapped)
    _pretrained_cache_patched = True


def get_f5_generate():
    """Return f5_tts_mlx.generate.generate with compatibility patches applied."""
    global _generate
    if _generate is not None:
        return _generate

    _f5_utils.lens_to_mask = _lens_to_mask_compat
    from f5_tts_mlx.generate import generate

    import f5_tts_mlx.cfm as _f5_cfm
    import f5_tts_mlx.duration as _f5_duration

    _f5_cfm.lens_to_mask = _lens_to_mask_compat
    _f5_duration.lens_to_mask = _lens_to_mask_compat
    _patch_f5_pretrained_cache()
    _generate = generate
    return generate


def play_wav(path: Path) -> None:
    path = path.resolve()
    if sys.platform == "darwin":
        subprocess.run(["afplay", str(path)], check=True)
        return
    for cmd in (
        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)],
        ["aplay", str(path)],
    ):
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    print(f"[F5] Install `ffplay` (ffmpeg) or use macOS `afplay` to hear: {path}", flush=True)


def speak_f5(
    text: str,
    *,
    ref_audio: str = "ref.wav",
    ref_text: str = "This is the reference voice speaking.",
    hf_model: str = "alandao/f5-tts-mlx-4bit",
    steps: int = 8,
) -> None:
    """Synthesize to a temp WAV and play (no sounddevice stream)."""
    generate = get_f5_generate()
    ref_path = ref_audio if ref_audio and Path(ref_audio).is_file() else None
    ref_t = ref_text if ref_path else None
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav = tmp.name
    try:
        generate(
            generation_text=text,
            model_name=hf_model,
            ref_audio_path=ref_path,
            ref_audio_text=ref_t,
            steps=steps,
            output_path=wav,
        )
        play_wav(Path(wav))
    finally:
        try:
            os.unlink(wav)
        except OSError:
            pass
