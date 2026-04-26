"""Fast local TTS via hexgrad/kokoro (Kokoro-82M)."""

from __future__ import annotations

import os
import subprocess
import tempfile
from contextlib import nullcontext
from pathlib import Path
import threading

import numpy as np

_pipeline_cache: dict[str, object] = {}
_pipeline_lock = threading.Lock()


def play_wav(path: Path) -> None:
    path = path.resolve()
    if os.sys.platform == "darwin":
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
    raise RuntimeError("No audio player found (tried afplay/ffplay/aplay).")


def _to_numpy(audio) -> np.ndarray:
    try:
        import torch

        if isinstance(audio, torch.Tensor):
            return audio.detach().cpu().float().numpy().reshape(-1)
    except ImportError:
        pass
    return np.asarray(audio, dtype=np.float32).reshape(-1)


def _inference_ctx():
    try:
        import torch

        return torch.inference_mode()
    except ImportError:
        return nullcontext()


def get_kokoro_pipeline(lang_code: str = "a"):
    try:
        from kokoro import KPipeline
    except ImportError as e:
        raise RuntimeError(
            "Missing kokoro. Install with:\n"
            "  pip install 'kokoro>=0.9.4' soundfile torch\n"
            "macOS English also needs:\n"
            "  brew install espeak-ng\n"
            "Optional GPU on Apple Silicon:\n"
            "  PYTORCH_ENABLE_MPS_FALLBACK=1"
        ) from e
    with _pipeline_lock:
        if lang_code not in _pipeline_cache:
            _pipeline_cache[lang_code] = KPipeline(
                lang_code=lang_code, repo_id="hexgrad/Kokoro-82M"
            )
        return _pipeline_cache[lang_code]


def prewarm_kokoro(*, voice: str = "af_heart", lang_code: str = "a", speed: float = 1.0) -> None:
    """Warm model + first kernel path with a tiny utterance."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav = tmp.name
    try:
        synthesize_kokoro_to_wav(
            "ok", wav, voice=voice, lang_code=lang_code, speed=speed
        )
    finally:
        try:
            os.unlink(wav)
        except OSError:
            pass


def synthesize_kokoro_to_wav(
    text: str,
    wav_path: str | Path,
    *,
    voice: str = "af_heart",
    lang_code: str = "a",
    speed: float = 1.0,
) -> None:
    """Run Kokoro forward only; write 24 kHz WAV to ``wav_path`` (for pipelined play)."""
    text = (text or "").strip()
    if not text:
        raise ValueError("empty text")

    pipeline = get_kokoro_pipeline(lang_code)
    import soundfile as sf

    pieces: list[np.ndarray] = []
    with _inference_ctx():
        for _, _, audio in pipeline(text, voice=voice, speed=speed):
            pieces.append(_to_numpy(audio))
    if not pieces:
        raise RuntimeError("Kokoro produced no audio")
    full = np.concatenate(pieces) if len(pieces) > 1 else pieces[0]
    sf.write(str(wav_path), full, 24000)


def speak_kokoro(
    text: str,
    *,
    voice: str = "af_heart",
    lang_code: str = "a",
    speed: float = 1.0,
) -> None:
    """Synthesize `text` with Kokoro, write 24 kHz WAV, play with afplay/ffplay."""
    text = (text or "").strip()
    if not text:
        return

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav = tmp.name
    try:
        synthesize_kokoro_to_wav(
            text, wav, voice=voice, lang_code=lang_code, speed=speed
        )
        play_wav(Path(wav))
    finally:
        try:
            os.unlink(wav)
        except OSError:
            pass
