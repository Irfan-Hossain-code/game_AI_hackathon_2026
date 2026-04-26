from __future__ import annotations

import os
import tempfile
import threading
import time

_whisper_model = None


def get_whisper_model(model_name: str):
    """Load and cache faster-whisper model once per process."""
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel

        _whisper_model = WhisperModel(model_name, device="cpu", compute_type="int8")
    return _whisper_model


def hold_to_talk_to_text(
    *,
    hold_key: str,
    model_name: str,
    language: str,
    max_seconds: float = 20.0,
) -> str:
    """Hold key to record; release to stop; transcribe with faster-whisper."""
    import numpy as np
    import sounddevice as sd
    import soundfile as sf
    from pynput import keyboard

    sample_rate = 16_000
    key_down = threading.Event()
    key_released = threading.Event()

    def _matches(key) -> bool:
        hk = (hold_key or "").strip().lower() or "space"
        if hk == "space":
            return key == keyboard.Key.space
        if hk == "enter":
            return key == keyboard.Key.enter
        if hk == "shift":
            return key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r)
        if hk == "ctrl":
            return key in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r)
        if hk == "alt":
            return key in (keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r)
        if hasattr(key, "char") and key.char:
            return key.char.lower() == hk
        return False

    def on_press(key):
        if _matches(key):
            key_down.set()

    def on_release(key):
        if _matches(key):
            key_released.set()
            return False
        return None

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        while not key_down.wait(timeout=0.05):
            pass

        chunks = []

        def _cb(indata, frames, tinfo, status):
            _ = frames, tinfo, status
            chunks.append(indata.copy())

        with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32", callback=_cb):
            t0 = time.perf_counter()
            while not key_released.is_set():
                if time.perf_counter() - t0 > max_seconds:
                    break
                time.sleep(0.01)
        listener.stop()

    if not chunks:
        return ""

    audio = np.concatenate(chunks, axis=0).reshape(-1)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name
    try:
        sf.write(wav_path, audio, sample_rate)
        model = get_whisper_model(model_name)
        segments, _ = model.transcribe(
            wav_path,
            language=language or None,
            vad_filter=True,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()
    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass
