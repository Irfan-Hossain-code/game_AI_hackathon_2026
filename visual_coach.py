#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import os
import queue
import subprocess
import threading
import time

import cv2
import numpy as np
import requests

from kokoro_speech import speak_kokoro


def adb_screencap(timeout_s: float) -> np.ndarray | None:
    try:
        proc = subprocess.run(
            ["adb", "exec-out", "screencap", "-p"],
            capture_output=True,
            check=False,
            timeout=timeout_s,
        )
        if proc.returncode != 0 or not proc.stdout:
            return None
        arr = np.frombuffer(proc.stdout, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def mss_screencap(sct, monitor_index: int) -> np.ndarray | None:
    try:
        monitors = sct.monitors
        max_idx = len(monitors) - 1
        idx = max(1, min(monitor_index, max_idx)) if max_idx >= 1 else 0
        mon = monitors[idx]
        shot = sct.grab(mon)
        frame = np.array(shot, dtype=np.uint8)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    except Exception:
        return None


def encode_frame(frame: np.ndarray, max_side: int = 720, jpg_quality: int = 55) -> str:
    h, w = frame.shape[:2]
    side = max(h, w)
    if side > max_side:
        scale = max_side / float(side)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def fetch_advice(frame: np.ndarray, *, host: str, model: str, timeout_s: float, max_tokens: int) -> str:
    b64 = encode_frame(frame)
    if not b64:
        return ""
    h, w = frame.shape[:2]
    prompt = (
        "You are a live Clash Royale coach. "
        "Look at this screenshot and give exactly one short spoken coaching sentence. "
        "Max 14 words. Clear and direct. No markdown. No JSON. "
        "If this is not an active battle, say exactly what to do next in menu. "
        f"Image size: {w}x{h}."
    )
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "messages": [{"role": "user", "content": prompt, "images": [b64]}],
        "options": {"temperature": 0.2, "num_predict": max_tokens},
    }
    try:
        r = requests.post(f"{host.rstrip('/')}/api/chat", json=payload, timeout=timeout_s)
        r.raise_for_status()
        text = r.json().get("message", {}).get("content", "")
        return " ".join((text or "").strip().split())
    except Exception:
        return ""


def speaker_worker(q: queue.Queue[str], voice: str, speed: float) -> None:
    while True:
        line = q.get()
        if line == "__STOP__":
            q.task_done()
            return
        try:
            speak_kokoro(line, voice=voice, speed=speed)
        except Exception as exc:
            print(f"[VOICE] {exc}", flush=True)
        q.task_done()


def main() -> int:
    ap = argparse.ArgumentParser(description="Visual-only AI coach: watch screen and speak advice.")
    ap.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    ap.add_argument("--model", default=os.environ.get("VISION_MODEL", "gemma4:e4b"))
    ap.add_argument("--interval-s", type=float, default=2.0, help="Seconds between model advice requests")
    ap.add_argument("--timeout-s", type=float, default=35.0, help="Model request timeout")
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--voice", default=os.environ.get("KOKORO_VOICE", "af_bella"))
    ap.add_argument("--voice-speed", type=float, default=1.0)
    ap.add_argument("--preview", action="store_true", help="Show live preview window")
    ap.add_argument("--capture-backend", choices=["auto", "mss", "adb"], default="auto")
    ap.add_argument("--monitor-index", type=int, default=1)
    ap.add_argument("--adb-timeout-s", type=float, default=1.0)
    args = ap.parse_args()

    mss_ok = False
    sct = None
    if args.capture_backend in ("auto", "mss"):
        try:
            import mss  # type: ignore

            sct = mss.mss()
            mss_ok = True
        except Exception:
            mss_ok = False

    q: queue.Queue[str] = queue.Queue(maxsize=1)
    t = threading.Thread(target=speaker_worker, args=(q, args.voice, args.voice_speed), daemon=True)
    t.start()

    last_advice_t = 0.0
    last_spoken = ""
    last_frame = None
    print(f"[COACH] model={args.model} interval={args.interval_s}s backend={args.capture_backend}", flush=True)

    try:
        while True:
            frame = None
            used = "none"
            if args.capture_backend == "adb":
                frame = adb_screencap(args.adb_timeout_s)
                used = "adb"
            elif args.capture_backend == "mss":
                if mss_ok and sct is not None:
                    frame = mss_screencap(sct, args.monitor_index)
                    used = "mss"
            else:
                if mss_ok and sct is not None:
                    frame = mss_screencap(sct, args.monitor_index)
                    used = "mss"
                if frame is None:
                    frame = adb_screencap(args.adb_timeout_s)
                    used = "adb"

            if frame is not None:
                last_frame = frame

            if args.preview:
                canvas = frame if frame is not None else np.zeros((480, 854, 3), dtype=np.uint8)
                cv2.putText(
                    canvas,
                    f"Visual Coach ({used}) - q to quit",
                    (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("Visual Coach", canvas)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

            now = time.perf_counter()
            if last_frame is not None and (now - last_advice_t) >= max(0.4, args.interval_s):
                t0 = time.perf_counter()
                advice = fetch_advice(
                    last_frame,
                    host=args.host,
                    model=args.model,
                    timeout_s=args.timeout_s,
                    max_tokens=args.max_tokens,
                )
                print(f"[LAT] advice={(time.perf_counter() - t0) * 1000:.1f}ms", flush=True)
                if advice and advice.lower() != last_spoken.lower():
                    print(f"[COACH] {advice}", flush=True)
                    if q.empty():
                        q.put_nowait(advice)
                    last_spoken = advice
                last_advice_t = now

            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        if args.preview:
            cv2.destroyAllWindows()
        try:
            if q.empty():
                q.put_nowait("__STOP__")
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
