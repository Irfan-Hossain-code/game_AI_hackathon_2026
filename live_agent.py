#!/usr/bin/env python3
"""
Stream an answer from Ollama, optionally speak with Kokoro (fast) or F5-TTS (clone quality).

- Kokoro: pip install 'kokoro>=0.9.4' soundfile torch  &&  brew install espeak-ng  (macOS)
  Streaming Kokoro: text chunks after ~`--kokoro-min-chars`; a synth thread writes WAVs
  while a separate thread plays them (next chunk can render during playback).
- F5: WAV + afplay; sentence pipelining unless *-one-shot.

Examples:
  python3 live_agent.py "What is 2+2?"
  PYTORCH_ENABLE_MPS_FALLBACK=1 python3 live_agent.py "who is khabib nurmagomedov?" --kokoro
  ./run_live.sh "who is khabib nurmagomedov?" --kokoro
  # Default LLM is gemma2:2b; override: OLLAMA_MODEL=gemma4:e4b ./run_live.sh "…" --kokoro
  python3 live_agent.py "Long question" --kokoro --kokoro-one-shot
  python3 live_agent.py "Long question" --f5
"""

from __future__ import annotations

import argparse
from collections import deque
import json
import os
import queue
import tempfile
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Optional


STOP_TTS = object()
_whisper_model = None


def _tts_log(msg: str) -> None:
    """TTS progress on stderr so stdout stays a clean streamed answer."""
    print(msg, file=sys.stderr, flush=True)


def get_whisper_model(model_name: str):
    """Load and cache faster-whisper model once per process."""
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel

        _whisper_model = WhisperModel(model_name, device="cpu", compute_type="int8")
    return _whisper_model


def mic_to_text(
    *,
    seconds: float,
    model_name: str,
    language: str,
) -> str:
    """Record from mic, transcribe with faster-whisper, return text."""
    import sounddevice as sd
    import soundfile as sf

    sample_rate = 16_000
    _tts_log(f"[STT] recording {seconds:.1f}s…")
    audio = sd.rec(
        int(seconds * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
    )
    sd.wait()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name
    try:
        sf.write(wav_path, audio.reshape(-1), sample_rate)
        model = get_whisper_model(model_name)
        segments, _ = model.transcribe(
            wav_path,
            language=language or None,
            vad_filter=True,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        _tts_log(f"[STT] -> {text or '(no speech detected)'}")
        return text
    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass


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
        hk = (hold_key or "").strip().lower()
        if not hk:
            hk = "space"
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

    _tts_log(f"[STT] hold '{hold_key}' to talk (release to stop)…")
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        # wait for first press
        while not key_down.wait(timeout=0.05):
            pass
        _tts_log("[STT] recording…")
        chunks = []

        def _cb(indata, frames, tinfo, status):
            _ = frames, tinfo
            if status:
                pass
            chunks.append(indata.copy())

        with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32", callback=_cb):
            t0 = time.perf_counter()
            while not key_released.is_set():
                if time.perf_counter() - t0 > max_seconds:
                    _tts_log("[STT] max record time reached; stopping.")
                    break
                time.sleep(0.01)
        listener.stop()

    if not chunks:
        _tts_log("[STT] -> (no speech detected)")
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
        text = " ".join(seg.text.strip() for seg in segments).strip()
        _tts_log(f"[STT] -> {text or '(no speech detected)'}")
        return text
    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass


class MutableDequeQueue:
    """Simple mutable FIFO with blocking pop and direct append."""

    def __init__(self) -> None:
        self._dq = deque()
        self._closed = False
        self._seq = 0
        self._cv = threading.Condition()

    def append(self, item) -> None:
        with self._cv:
            if self._closed:
                return
            self._dq.append((self._seq, item))
            self._seq += 1
            self._cv.notify()

    def put(self, item) -> None:
        self.append(item)

    def pop(self):
        with self._cv:
            while not self._dq and not self._closed:
                self._cv.wait()
            if self._dq:
                return self._dq.popleft()
            return STOP_TTS

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()


def stream_ollama_chat(
    host: str,
    model: str,
    user_text: str,
    system_prompt: str | None,
    on_token_chunk: Optional[Callable[[str], None]] = None,
) -> tuple[str, float, float]:
    """
    Stream /api/chat. Optional on_token_chunk for live side effects.
    Returns (full_reply, seconds_to_first_token, seconds_total_llm).
    """
    host = host.rstrip("/")
    url = f"{host}/api/chat"
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_text})
    body = json.dumps(
        {"model": model, "messages": messages, "stream": True}
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    full: list[str] = []
    t_start = time.perf_counter()
    t_first: float | None = None

    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            while True:
                raw = resp.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("error"):
                    raise RuntimeError(obj["error"])
                msg = obj.get("message") or {}
                chunk = msg.get("content") or ""
                if chunk:
                    if t_first is None:
                        t_first = time.perf_counter()
                    full.append(chunk)
                    print(chunk, end="", flush=True)
                    if on_token_chunk is not None:
                        on_token_chunk(chunk)
                if obj.get("done"):
                    break
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Cannot reach Ollama at {url!r}. Is `ollama serve` running? ({e})"
        ) from e

    print()
    text = "".join(full).strip()
    t_end = time.perf_counter()
    ttft = (t_first - t_start) if t_first is not None else float("nan")
    return text, ttft, t_end - t_start


def _flush_sentence_buffer(buf: str, q: queue.Queue) -> str:
    """Move complete sentences from buf into q; return remaining tail."""
    while True:
        parts = re.split(r"(?<=[.!?])\s+", buf)
        if len(parts) < 2:
            return buf
        for s in parts[:-1]:
            s = s.strip()
            if s:
                q.put(s)
        buf = parts[-1]


def _kokoro_flush_token_chunks(
    buf: str, q: queue.Queue, *, min_chars: int, max_chars: int
) -> str:
    """Enqueue Kokoro-sized chunks after ~min_chars (break at last space); avoid huge waits."""
    rest = buf
    while True:
        if len(rest) < min_chars:
            return rest
        window = min(len(rest), min_chars + 72)
        head = rest[:window]
        br = max(head.rfind(" "), head.rfind("\n"))
        if br > 0:
            piece = rest[:br].strip()
            rest = rest[br + 1 :]
            if piece:
                q.put(piece)
            continue
        if len(rest) >= max_chars:
            head2 = rest[:max_chars]
            br2 = max(head2.rfind(" "), head2.rfind("\n"))
            if br2 > 8:
                piece = rest[:br2].strip()
                rest = rest[br2 + 1 :]
            else:
                piece = rest[:max_chars].strip()
                rest = rest[max_chars:]
            if piece:
                q.put(piece)
            continue
        return rest


def _make_kokoro_chunk_flush(min_chars: int, max_chars: int) -> Callable[[str, queue.Queue], str]:
    def flush(buf: str, q: queue.Queue) -> str:
        return _kokoro_flush_token_chunks(buf, q, min_chars=min_chars, max_chars=max_chars)

    return flush


def _run_llm_with_streaming_tts(
    worker_target: Callable[..., None],
    worker_kwargs: dict,
    host: str,
    model: str,
    q_text: str,
    system: str | None,
    buffer_flush: Callable[[str, queue.Queue], str],
) -> tuple[str, float, float]:
    tts_q: queue.Queue = queue.Queue()
    worker = threading.Thread(
        target=worker_target,
        args=(tts_q,),
        kwargs=worker_kwargs,
        daemon=True,
    )
    worker.start()
    buf_holder: list[str] = [""]

    def on_chunk(ch: str) -> None:
        buf_holder[0] += ch
        buf_holder[0] = buffer_flush(buf_holder[0], tts_q)

    reply, ttft, llm_dt = stream_ollama_chat(
        host, model, q_text, system, on_token_chunk=on_chunk
    )
    buf_holder[0] = buffer_flush(buf_holder[0], tts_q)
    if buf_holder[0].strip():
        tts_q.put(buf_holder[0].strip())
    tts_q.put(STOP_TTS)
    worker.join()
    return reply, ttft, llm_dt


def run_f5_worker(
    q: queue.Queue,
    *,
    ref_audio: str,
    ref_text: str,
    hf_model: str,
    steps: int,
) -> None:
    from f5_mlx_compat import speak_f5

    while True:
        item = q.get()
        if item is STOP_TTS:
            break
        if not isinstance(item, str) or not item.strip():
            continue
        _tts_log(f"\n[F5] sentence ({len(item)} chars)…")
        t0 = time.perf_counter()
        try:
            speak_f5(
                item.strip(),
                ref_audio=ref_audio,
                ref_text=ref_text,
                hf_model=hf_model,
                steps=steps,
            )
        except subprocess.CalledProcessError as e:
            _tts_log(f"[F5] playback failed: {e}")
        _tts_log(f"[F5] done in {time.perf_counter() - t0:.2f} s")


def run_kokoro_synth_worker(
    text_q: MutableDequeQueue,
    audio_q: queue.Queue,
    *,
    worker_id: int,
    voice: str,
    lang_code: str,
    speed: float,
) -> None:
    """Text chunks -> WAV paths (Kokoro forward pass only)."""
    from kokoro_speech import synthesize_kokoro_to_wav

    while True:
        item = text_q.pop()
        if item is STOP_TTS:
            break
        if not isinstance(item, tuple) or len(item) != 2:
            continue
        seq, text = item
        if not isinstance(text, str) or not text.strip():
            continue
        text = text.strip()
        _tts_log(f"\n[Kokoro] synth#{worker_id} chunk {seq} ({len(text)} chars)…")
        t0 = time.perf_counter()
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            synthesize_kokoro_to_wav(
                text, path, voice=voice, lang_code=lang_code, speed=speed
            )
            audio_q.put((seq, path))
        except Exception as e:
            _tts_log(f"[Kokoro] synth failed: {e!r}")
            try:
                os.unlink(path)
            except OSError:
                pass
        else:
            _tts_log(f"[Kokoro] synth#{worker_id} ok {seq} in {time.perf_counter() - t0:.2f} s")
    audio_q.put(STOP_TTS)


def run_kokoro_play_worker(audio_q: queue.Queue, synth_workers: int) -> None:
    """Play WAV paths in-order while synth workers run ahead."""
    from f5_mlx_compat import play_wav
    import soundfile as sf
    import numpy as np

    done_workers = 0
    next_seq = 0
    pending: dict[int, str] = {}

    def _play_batch(paths: list[str]) -> None:
        if len(paths) == 1:
            play_wav(Path(paths[0]))
            return
        chunks = []
        sr_ref = 24000
        for p in paths:
            audio, sr = sf.read(p)
            if sr != sr_ref:
                sr_ref = sr
            chunks.append(np.asarray(audio, dtype=np.float32).reshape(-1))
        merged = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        fd, merged_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            sf.write(merged_path, merged, sr_ref)
            play_wav(Path(merged_path))
        finally:
            try:
                os.unlink(merged_path)
            except OSError:
                pass

    while True:
        item = audio_q.get()
        if item is STOP_TTS:
            done_workers += 1
        elif isinstance(item, tuple) and len(item) == 2:
            seq, path = item
            if isinstance(seq, int) and isinstance(path, str):
                pending[seq] = path

        while next_seq in pending:
            batch = []
            while next_seq in pending and len(batch) < 3:
                batch.append(pending.pop(next_seq))
                next_seq += 1
            _tts_log("[Kokoro] playing chunk…")
            t0 = time.perf_counter()
            try:
                _play_batch(batch)
            except subprocess.CalledProcessError as e:
                _tts_log(f"[Kokoro] play failed: {e}")
            finally:
                for p in batch:
                    try:
                        os.unlink(p)
                    except OSError:
                        pass
            _tts_log(f"[Kokoro] play ok in {time.perf_counter() - t0:.2f} s")

        if done_workers >= synth_workers and not pending:
            break


def _run_llm_with_kokoro_pipeline_tts(
    host: str,
    model: str,
    q_text: str,
    system: str | None,
    buffer_flush: Callable[[str, object], str],
    *,
    synth_workers: int,
    voice: str,
    lang_code: str,
    speed: float,
) -> tuple[str, float, float]:
    from kokoro_speech import prewarm_kokoro

    text_q = MutableDequeQueue()
    audio_q: queue.Queue = queue.Queue()
    synth_threads = []
    for wid in range(synth_workers):
        th = threading.Thread(
            target=run_kokoro_synth_worker,
            args=(text_q, audio_q),
            kwargs={
                "worker_id": wid + 1,
                "voice": voice,
                "lang_code": lang_code,
                "speed": speed,
            },
            daemon=True,
        )
        synth_threads.append(th)
    player = threading.Thread(
        target=run_kokoro_play_worker,
        args=(audio_q, synth_workers),
        daemon=True,
    )
    for th in synth_threads:
        th.start()
    player.start()
    warm = threading.Thread(
        target=prewarm_kokoro,
        kwargs={"voice": voice, "lang_code": lang_code, "speed": speed},
        daemon=True,
    )
    warm.start()

    buf_holder: list[str] = [""]

    def on_chunk(ch: str) -> None:
        buf_holder[0] += ch
        buf_holder[0] = buffer_flush(buf_holder[0], text_q)

    reply, ttft, llm_dt = stream_ollama_chat(
        host, model, q_text, system, on_token_chunk=on_chunk
    )
    buf_holder[0] = buffer_flush(buf_holder[0], text_q)
    if buf_holder[0].strip():
        text_q.append(buf_holder[0].strip())
    text_q.close()
    for th in synth_threads:
        th.join()
    warm.join()
    player.join()
    return reply, ttft, llm_dt


def main() -> int:
    p = argparse.ArgumentParser(description="Ollama streaming agent (+ optional Kokoro or F5-TTS).")
    p.add_argument("question", nargs="?", help="Question to ask (or use -q)")
    p.add_argument("-q", "--question", dest="q_alt", help="Question text")
    p.add_argument(
        "--loop",
        action="store_true",
        help="Keep process alive; ask multiple questions in one hot session.",
    )
    p.add_argument(
        "--stt",
        action="store_true",
        help="Enable speech-to-text input in loop mode (press Enter to record).",
    )
    p.add_argument(
        "--stt-seconds",
        type=float,
        default=3.5,
        help="Mic capture duration for one STT query (default 3.5s).",
    )
    p.add_argument(
        "--stt-model",
        default="tiny.en",
        help="faster-whisper model (default tiny.en, faster; try small for better accuracy).",
    )
    p.add_argument(
        "--stt-lang",
        default="en",
        help="Whisper language code (default en).",
    )
    p.add_argument(
        "--stt-hold-key",
        default="space",
        help="Push-to-talk key for STT in loop mode (space, enter, shift, ctrl, alt, or letter).",
    )
    p.add_argument(
        "--host",
        default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
        help="Ollama base URL (default OLLAMA_HOST or http://127.0.0.1:11434)",
    )
    p.add_argument(
        "--model",
        default=os.environ.get("OLLAMA_MODEL", "gemma2:2b"),
        help="Ollama model name (default OLLAMA_MODEL or gemma2:2b)",
    )
    p.add_argument(
        "--system",
        default=(
            "You are a concise assistant. Use the fewest words that stay accurate. "
            "No preamble, no meta-commentary, no bullet lists unless asked."
        ),
        help="System prompt",
    )
    p.add_argument(
        "--kokoro",
        action="store_true",
        help="Speak with Kokoro (fast). Synth+play threads pipeline chunks (see --kokoro-min-chars).",
    )
    p.add_argument(
        "--kokoro-synth-workers",
        type=int,
        default=2,
        metavar="N",
        help="Number of Kokoro synthesis workers (default 2).",
    )
    p.add_argument(
        "--kokoro-extra-play-threads",
        type=int,
        default=0,
        metavar="N",
        help="Reserved knob for future tuning. Playback stays single-threaded to preserve audio order.",
    )
    p.add_argument(
        "--kokoro-one-shot",
        action="store_true",
        help="Speak the full reply once after streaming (Kokoro).",
    )
    p.add_argument(
        "--kokoro-wait-sentences",
        action="store_true",
        help="Kokoro: wait for .!? sentence boundaries instead of early character chunks.",
    )
    p.add_argument(
        "--kokoro-min-chars",
        type=int,
        default=40,
        metavar="N",
        help="Kokoro streaming: enqueue after ~N chars (break at last space; default 40).",
    )
    p.add_argument(
        "--kokoro-max-chars",
        type=int,
        default=180,
        metavar="N",
        help="Kokoro streaming: hard-split if no space before N chars (default 180).",
    )
    p.add_argument("--kokoro-voice", default="af_heart", help="Kokoro voice id (default af_heart).")
    p.add_argument("--kokoro-lang", default="a", help="Kokoro lang_code: a=US English, b=British, …")
    p.add_argument(
        "--kokoro-speed",
        type=float,
        default=1.12,
        help="Kokoro playback speed (>1 faster speech; default 1.12).",
    )
    p.add_argument(
        "--f5",
        action="store_true",
        help="Speak with F5-TTS (slower, voice clone). Default: sentence pipelining.",
    )
    p.add_argument(
        "--f5-one-shot",
        action="store_true",
        help="Synthesize the full reply once after streaming (F5).",
    )
    p.add_argument("--f5-steps", type=int, default=8, help="F5 sampler steps (default 8).")
    p.add_argument(
        "--f5-hf-model",
        default="alandao/f5-tts-mlx-4bit",
        help="Hugging Face id for f5_tts_mlx weights",
    )
    p.add_argument("--ref-audio", default="ref.wav", help="24 kHz ref WAV (optional, F5 only)")
    p.add_argument(
        "--ref-text",
        default="This is the reference voice speaking.",
        help="Transcript of ref-audio when using a custom ref (F5 only)",
    )
    args = p.parse_args()

    if args.f5 and args.kokoro:
        p.error("Use either --f5 or --kokoro, not both.")
    if args.kokoro_max_chars < args.kokoro_min_chars:
        p.error("--kokoro-max-chars must be >= --kokoro-min-chars")
    if args.kokoro_synth_workers < 1:
        p.error("--kokoro-synth-workers must be >= 1")
    if args.stt_seconds <= 0:
        p.error("--stt-seconds must be > 0")
    if args.kokoro_extra_play_threads:
        _tts_log(
            "[Kokoro] note: extra playback threads are disabled to keep chunk order and avoid overlapping speech."
        )

    print(f"Model: {args.model!r}  host: {args.host!r}\n", flush=True)
    def run_one_question(q_text: str) -> None:
        print("--- answer (streaming) ---", flush=True)
        t_wall0 = time.perf_counter()

        if args.kokoro and not args.kokoro_one_shot:
            if args.kokoro_wait_sentences:
                kok_flush: Callable[[str, queue.Queue], str] = _flush_sentence_buffer
            else:
                kok_flush = _make_kokoro_chunk_flush(
                    args.kokoro_min_chars,
                    args.kokoro_max_chars,
                )
            reply, ttft, llm_dt = _run_llm_with_kokoro_pipeline_tts(
                args.host,
                args.model,
                q_text,
                args.system or None,
                buffer_flush=kok_flush,
                synth_workers=args.kokoro_synth_workers,
                voice=args.kokoro_voice,
                lang_code=args.kokoro_lang,
                speed=args.kokoro_speed,
            )
        elif args.f5 and not args.f5_one_shot:
            if not Path(args.ref_audio).is_file() and args.ref_audio:
                _tts_log(
                    f"\n[F5] No file at {args.ref_audio!r} — using built-in reference audio."
                )
            reply, ttft, llm_dt = _run_llm_with_streaming_tts(
                run_f5_worker,
                {
                    "ref_audio": args.ref_audio,
                    "ref_text": args.ref_text,
                    "hf_model": args.f5_hf_model,
                    "steps": args.f5_steps,
                },
                args.host,
                args.model,
                q_text,
                args.system or None,
                buffer_flush=_flush_sentence_buffer,
            )
        else:
            reply, ttft, llm_dt = stream_ollama_chat(
                args.host, args.model, q_text, args.system or None, on_token_chunk=None
            )

        print("--- end stream ---", flush=True)
        print(
            f"\n[time] first token: {ttft*1000:.0f} ms  |  LLM total: {llm_dt*1000:.0f} ms",
            flush=True,
        )

        if args.kokoro and args.kokoro_one_shot:
            if not reply:
                _tts_log("[Kokoro] Empty reply, skipping TTS.")
            else:
                _tts_log(f"\n[Kokoro] one-shot: {len(reply)} chars…")
                from kokoro_speech import speak_kokoro

                t0 = time.perf_counter()
                try:
                    speak_kokoro(
                        reply,
                        voice=args.kokoro_voice,
                        lang_code=args.kokoro_lang,
                        speed=args.kokoro_speed,
                    )
                except subprocess.CalledProcessError as e:
                    _tts_log(f"[Kokoro] playback failed: {e}")
                except RuntimeError as e:
                    _tts_log(f"[Kokoro] {e}")
                _tts_log(f"[Kokoro] done in {time.perf_counter() - t0:.2f} s")

        if args.f5 and args.f5_one_shot:
            if not reply:
                _tts_log("[F5] Empty reply, skipping TTS.")
            else:
                if not Path(args.ref_audio).is_file() and args.ref_audio:
                    _tts_log(
                        f"\n[F5] No file at {args.ref_audio!r} — using built-in reference audio."
                    )
                _tts_log(f"\n[F5] one-shot: {len(reply)} chars (steps={args.f5_steps})…")
                from f5_mlx_compat import speak_f5

                t0 = time.perf_counter()
                try:
                    speak_f5(
                        reply,
                        ref_audio=args.ref_audio,
                        ref_text=args.ref_text,
                        hf_model=args.f5_hf_model,
                        steps=args.f5_steps,
                    )
                except subprocess.CalledProcessError as e:
                    _tts_log(f"[F5] playback failed: {e}")
                _tts_log(f"[F5] done in {time.perf_counter() - t0:.2f} s")

        print(f"[time] wall total: {(time.perf_counter() - t_wall0)*1000:.0f} ms", flush=True)

    initial_q = args.question or args.q_alt
    if args.loop:
        if initial_q:
            run_one_question(initial_q)
        while True:
            try:
                if args.stt:
                    next_q = input(
                        "\nQuestion (type, Enter=push-to-talk mic, or 'exit'): "
                    ).strip()
                else:
                    next_q = input("\nQuestion (or 'exit'): ").strip()
            except EOFError:
                break
            if not next_q:
                if args.stt:
                    try:
                        next_q = hold_to_talk_to_text(
                            hold_key=args.stt_hold_key,
                            model_name=args.stt_model,
                            language=args.stt_lang,
                        )
                    except Exception as e:
                        _tts_log(f"[STT] hold-to-talk unavailable ({e!r}); using fixed record.")
                        next_q = mic_to_text(
                            seconds=args.stt_seconds,
                            model_name=args.stt_model,
                            language=args.stt_lang,
                        )
                else:
                    continue
            if not next_q:
                continue
            if next_q.lower() in {"exit", "quit", "q"}:
                break
            run_one_question(next_q)
    else:
        q_text = initial_q
        if not q_text:
            if sys.stdin.isatty():
                if args.stt:
                    q_text = mic_to_text(
                        seconds=args.stt_seconds,
                        model_name=args.stt_model,
                        language=args.stt_lang,
                    )
                else:
                    q_text = input("Question: ").strip()
            else:
                q_text = sys.stdin.read().strip()
        if not q_text:
            p.error("Need a question (argument, -q, or stdin).")
        run_one_question(q_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
