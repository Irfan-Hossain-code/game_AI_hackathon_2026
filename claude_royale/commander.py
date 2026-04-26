#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kokoro_speech import speak_kokoro
from screen_classifier import ScreenClassifier
from vision import MuMuController


@dataclass
class WorkerAction:
    should_play: bool
    slot: int
    target: str
    why: str
    confidence: float


class VisionBrain:
    def __init__(self, model: str, host: str, timeout_s: float) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout_s = timeout_s
        self.req_lock = threading.Lock()
        self.last_err_t = 0.0

    def _encode_frame(self, frame: np.ndarray) -> str:
        h, w = frame.shape[:2]
        side = max(h, w)
        if side > 720:
            scale = 720.0 / float(side)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 55])
        if not ok:
            return ""
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    def decide(self, frame: np.ndarray, worker_id: int, targets: dict[str, list[int]]) -> WorkerAction:
        b64 = self._encode_frame(frame)
        if not b64:
            return WorkerAction(False, 0, "center_defense", "capture_failed", 0.0)
        prompt = (
            "You are a Clash Royale player sub-agent.\n"
            "Return strict JSON only.\n"
            'Schema: {"should_play":true|false,"slot":0-3,"target":"name","why":"short","confidence":0.0-1.0}\n'
            f"worker_id={worker_id}\n"
            f"allowed_targets={json.dumps(sorted(targets.keys()), ensure_ascii=True)}\n"
            "Rules:\n"
            "- If no clear play, should_play=false.\n"
            "- If defending, use center_defense or back_defense target.\n"
            "- Keep why under 8 words.\n"
            "- No markdown.\n"
        )
        payload = {
            "model": self.model,
            "stream": False,
            "think": False,
            "messages": [{"role": "user", "content": prompt, "images": [b64]}],
            "options": {"temperature": 0.15, "num_predict": 80},
        }
        try:
            # Local multimodal models can choke on 3 parallel requests.
            # Serialize calls so workers still stagger card actions, not crashes/timeouts.
            with self.req_lock:
                r = requests.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout_s)
            r.raise_for_status()
            content = (r.json().get("message", {}).get("content", "") or "").strip()
            data: dict[str, Any] = {}
            try:
                data = json.loads(content)
            except Exception:
                return WorkerAction(False, 0, "center_defense", "parse_failed", 0.0)
            slot = int(data.get("slot", 0))
            target = str(data.get("target", "center_defense"))
            return WorkerAction(
                should_play=bool(data.get("should_play", False)),
                slot=max(0, min(3, slot)),
                target=target if target in targets else "center_defense",
                why=str(data.get("why", "tempo")),
                confidence=float(data.get("confidence", 0.0)),
            )
        except Exception as exc:
            now = time.perf_counter()
            if now - self.last_err_t > 2.5:
                print(f"[CR][LLM] worker={worker_id} error={type(exc).__name__}: {exc}", flush=True)
                self.last_err_t = now
            return WorkerAction(False, 0, "center_defense", "llm_failed", 0.0)


def _focused_app_package() -> str:
    """
    Best-effort foreground package detection from Android window manager.
    """
    try:
        proc = subprocess.run(
            ["adb", "shell", "dumpsys", "window", "windows"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        txt = (proc.stdout or "") + "\n" + (proc.stderr or "")
        for line in txt.splitlines():
            l = line.strip()
            if "mCurrentFocus" in l or "mFocusedApp" in l:
                # Typical tail format: com.supercell.clashroyale/com.supercell.titan.GameApp
                if "/" in l:
                    candidate = l.split("/")[-2] if " " not in l.split("/")[-2] else l
                # Fallback regex parse.
                import re

                m = re.search(r"([a-zA-Z0-9_.]+)\/[a-zA-Z0-9_.$]+", l)
                if m:
                    return m.group(1).strip()
        return ""
    except Exception:
        return ""


def _launch_game(package_name: str) -> bool:
    try:
        proc = subprocess.run(
            ["adb", "shell", "monkey", "-p", package_name, "1"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3.0,
        )
        return proc.returncode == 0
    except Exception:
        return False


class PlayerWorker(threading.Thread):
    def __init__(
        self,
        worker_id: int,
        *,
        brain: VisionBrain,
        controller: MuMuController,
        frame_ref: dict[str, np.ndarray | None],
        frame_lock: threading.Lock,
        tap_lock: threading.Lock,
        targets: dict[str, list[int]],
        action_q: queue.Queue[tuple[int, WorkerAction]],
        stop_event: threading.Event,
        interval_s: float,
        start_stagger_s: float,
    ) -> None:
        super().__init__(daemon=True)
        self.worker_id = worker_id
        self.brain = brain
        self.controller = controller
        self.frame_ref = frame_ref
        self.frame_lock = frame_lock
        self.tap_lock = tap_lock
        self.targets = targets
        self.action_q = action_q
        self.stop_event = stop_event
        self.interval_s = interval_s
        self.start_stagger_s = start_stagger_s
        self.last_action_t = 0.0
        self.last_status_t = 0.0
        self.last_fallback_t = 0.0

    def run(self) -> None:
        time.sleep(self.start_stagger_s)
        while not self.stop_event.is_set():
            now = time.perf_counter()
            if now - self.last_action_t < self.interval_s:
                time.sleep(0.02)
                continue
            with self.frame_lock:
                frame = self.frame_ref.get("frame")
                frame = None if frame is None else frame.copy()
            if frame is None:
                time.sleep(0.05)
                continue
            action = self.brain.decide(frame, self.worker_id, self.targets)
            if action.should_play:
                xy = self.targets.get(action.target)
                if xy:
                    with self.tap_lock:
                        ok = self.controller.play_card(action.slot, int(xy[0]), int(xy[1]))
                    if ok:
                        self.action_q.put((self.worker_id, action))
                    else:
                        if now - self.last_status_t > 2.5:
                            print(f"[CR][W{self.worker_id}] tap failed target={action.target}", flush=True)
                            self.last_status_t = now
            else:
                if now - self.last_status_t > 2.5:
                    print(
                        f"[CR][W{self.worker_id}] thinking... no_play why={action.why} conf={action.confidence:.2f}",
                        flush=True,
                    )
                    self.last_status_t = now
                # Safe fallback when multimodal inference repeatedly fails:
                # keep light pressure so bot doesn't freeze in battle.
                if action.why == "llm_failed" and (now - self.last_fallback_t) > 5.5:
                    fallback_slots = [0, 1, 2, 3]
                    fallback_targets = ["center_defense", "left_bridge", "right_bridge"]
                    slot = fallback_slots[self.worker_id % len(fallback_slots)]
                    tgt = fallback_targets[self.worker_id % len(fallback_targets)]
                    xy = self.targets.get(tgt)
                    if xy:
                        with self.tap_lock:
                            ok = self.controller.play_card(slot, int(xy[0]), int(xy[1]))
                        print(
                            f"[CR][W{self.worker_id}] fallback_play slot={slot} target={tgt} ok={int(ok)}",
                            flush=True,
                        )
                    self.last_fallback_t = now
            self.last_action_t = now


def _load_targets(gameplay_path: Path, fallback_anchors: dict[str, tuple[int, int]]) -> dict[str, list[int]]:
    if gameplay_path.is_file():
        try:
            data = json.loads(gameplay_path.read_text(encoding="utf-8"))
            targets = data.get("targets", {})
            out: dict[str, list[int]] = {}
            for k, v in targets.items():
                if isinstance(v, list) and len(v) == 2:
                    out[str(k)] = [int(v[0]), int(v[1])]
            if out:
                return out
        except Exception:
            pass
    return {
        "left_bridge": list(fallback_anchors.get("left_bridge", (390, 990))),
        "right_bridge": list(fallback_anchors.get("right_bridge", (690, 990))),
        "left_back_defense": list(fallback_anchors.get("left_back_defense", (300, 1290))),
        "right_back_defense": list(fallback_anchors.get("right_back_defense", (780, 1290))),
        "center_defense": list(fallback_anchors.get("center_defense", (540, 1240))),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Claude-Royale style commander with 3 parallel players.")
    ap.add_argument("--coords", default="mumu_coords.json")
    ap.add_argument("--gameplay", default="claude_royale/config/gameplay.json")
    ap.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    ap.add_argument("--player-model", default=os.environ.get("VISION_MODEL", "gemma4:e4b"))
    ap.add_argument("--player-timeout-s", type=float, default=25.0)
    ap.add_argument("--worker-interval-s", type=float, default=2.2)
    ap.add_argument("--battle-end-hold-s", type=float, default=2.8)
    ap.add_argument("--voice", action="store_true", help="Speak worker actions with Kokoro")
    ap.add_argument("--voice-name", default=os.environ.get("KOKORO_VOICE", "af_bella"))
    args = ap.parse_args()

    controller = MuMuController(coords_file=args.coords, dry_run=os.environ.get("DRY_RUN", "0") == "1")
    classifier = ScreenClassifier()
    brain = VisionBrain(model=args.player_model, host=args.host, timeout_s=args.player_timeout_s)
    targets = _load_targets(Path(args.gameplay), controller.anchors)
    package_name = os.environ.get("CR_PACKAGE", "com.supercell.clashroyale")
    action_q: queue.Queue[tuple[int, WorkerAction]] = queue.Queue()
    tap_lock = threading.Lock()
    frame_lock = threading.Lock()
    frame_ref: dict[str, np.ndarray | None] = {"frame": None}

    print("[CR] commander start", flush=True)
    print(f"[CR] model={args.player_model} workers=3 interval={args.worker_interval_s}s", flush=True)

    in_battle = False
    workers: list[PlayerWorker] = []
    stop_event = threading.Event()
    last_battle_seen_t = 0.0

    last_launch_t = 0.0
    last_heartbeat_t = 0.0
    while True:
        frame = controller.capture()
        with frame_lock:
            frame_ref["frame"] = frame
        s = classifier.classify(frame)
        battle_like = s.name == "in_battle" or controller.is_battle_view(frame)
        now = time.perf_counter()
        focused_pkg = _focused_app_package()

        if battle_like:
            last_battle_seen_t = now
            if not in_battle:
                in_battle = True
                stop_event.clear()
                workers = [
                    PlayerWorker(
                        i,
                        brain=brain,
                        controller=controller,
                        frame_ref=frame_ref,
                        frame_lock=frame_lock,
                        tap_lock=tap_lock,
                        targets=targets,
                        action_q=action_q,
                        stop_event=stop_event,
                        interval_s=args.worker_interval_s,
                        start_stagger_s=0.65 * i,
                    )
                    for i in range(3)
                ]
                for w in workers:
                    w.start()
                print("[CR] battle detected -> spawned 3 player workers", flush=True)
        else:
            if in_battle and (now - last_battle_seen_t) > args.battle_end_hold_s:
                stop_event.set()
                for w in workers:
                    w.join(timeout=1.0)
                workers = []
                in_battle = False
                print("[CR] battle ended -> workers stopped", flush=True)

            # Commander menu behavior, similar to claude-royale harness.
            if focused_pkg and package_name not in focused_pkg:
                if now - last_launch_t > 2.0:
                    ok = _launch_game(package_name)
                    print(f"[CR] relaunch outside app focused={focused_pkg or 'unknown'} ok={int(ok)}", flush=True)
                    last_launch_t = now
            elif s.name == "menu":
                controller.tap_anchor("battle_button")
            elif s.name == "popup":
                controller.tap_anchor("ok_button")
            else:
                # Relaunch only occasionally when outside known menu/popup/battle screens.
                if now - last_launch_t > 3.0:
                    _launch_game(package_name)
                    last_launch_t = now

        while not action_q.empty():
            worker_id, act = action_q.get()
            line = f"Worker {worker_id}: slot {act.slot} to {act.target}, {act.why}"
            print(f"[CR][PLAY] {line} conf={act.confidence:.2f}", flush=True)
            if args.voice:
                try:
                    speak_kokoro(line, voice=args.voice_name)
                except Exception as exc:
                    print(f"[CR][VOICE] {exc}", flush=True)

        if now - last_heartbeat_t > 3.0:
            print(
                f"[CR][HB] state={s.name} in_battle={int(in_battle)} workers={len(workers)} focused={focused_pkg or 'unknown'}",
                flush=True,
            )
            last_heartbeat_t = now

        time.sleep(0.08)


if __name__ == "__main__":
    raise SystemExit(main())
