from __future__ import annotations

import os
import time
import json
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

import cv2
import mss
import numpy as np


@dataclass
class ArenaState:
    ts: float
    frame_id: int
    enemy_cards_seen: list[str] = field(default_factory=list)
    enemy_elixir_est: int = 5
    enemy_log_available: bool = True
    teammate_context: str = ""
    signals: dict[str, Any] = field(default_factory=dict)


class MuMuController:
    """
    MuMu screen polling + ADB control.
    Coordinates are relative to the Android canvas.
    """

    def __init__(
        self,
        left: int = 0,
        top: int = 0,
        width: int = 1080,
        height: int = 1920,
        dry_run: bool = False,
        min_tap_ms: int = 110,
        coords_file: str = "mumu_coords.json",
    ) -> None:
        self.region = {"left": left, "top": top, "width": width, "height": height}
        self.canvas_width = width
        self.canvas_height = height
        self.dry_run = dry_run or os.environ.get("DRY_RUN", "0") == "1"
        self.use_adb_screencap = os.environ.get("USE_ADB_SCREENSHOT", "1") == "1"
        self.min_tap_s = min_tap_ms / 1000.0
        self.last_tap_t = 0.0
        self.sct = mss.mss()
        self.frame_id = 0

        self.enemy_cards_seen: set[str] = set()
        self.enemy_elixir_est = 5
        self.enemy_log_available = True
        self.last_enemy_spend_t = time.perf_counter()
        self.hand_slots = self._load_hand_slots(coords_file)
        self.anchors = self._load_anchors(coords_file)

    def capture(self) -> np.ndarray:
        arr: np.ndarray | None = None
        if self.use_adb_screencap:
            arr = self._capture_from_adb()
        if arr is None:
            shot = self.sct.grab(self.region)
            arr = np.array(shot)[:, :, :3]
        self.frame_id += 1
        return arr

    def _capture_from_adb(self) -> np.ndarray | None:
        """
        Read device framebuffer directly via ADB.
        This avoids desktop window-position dependency.
        """
        try:
            proc = subprocess.run(
                ["adb", "exec-out", "screencap", "-p"],
                capture_output=True,
                check=False,
            )
            if proc.returncode != 0 or not proc.stdout:
                return None
            buf = np.frombuffer(proc.stdout, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is None:
                return None
            # Normalize to configured canvas size for coordinate consistency.
            h, w = img.shape[:2]
            if w != self.canvas_width or h != self.canvas_height:
                img = cv2.resize(
                    img,
                    (self.canvas_width, self.canvas_height),
                    interpolation=cv2.INTER_AREA,
                )
            return img
        except Exception:
            return None

    def tap(self, x: int, y: int) -> bool:
        now = time.perf_counter()
        if now - self.last_tap_t < self.min_tap_s:
            return False
        self.last_tap_t = now

        x = max(0, min(self.canvas_width - 1, int(x)))
        y = max(0, min(self.canvas_height - 1, int(y)))
        cmd = f"adb shell input tap {x} {y}"
        if self.dry_run:
            print(f"[ADB][DRY] {cmd}", flush=True)
            return True
        return os.system(cmd) == 0

    def keyevent(self, key_code: int) -> bool:
        cmd = f"adb shell input keyevent {int(key_code)}"
        if self.dry_run:
            print(f"[ADB][DRY] {cmd}", flush=True)
            return True
        return os.system(cmd) == 0

    def back(self) -> bool:
        return self.keyevent(4)

    def play_card(self, slot: int, tx: int, ty: int) -> bool:
        if slot not in self.hand_slots:
            return False
        sx, sy = self.hand_slots[slot]
        ok1 = self.tap(sx, sy)
        ok2 = self.tap(tx, ty)
        return ok1 and ok2

    def _default_hand_slots(self) -> dict[int, tuple[int, int]]:
        # Relative positions that scale with canvas size.
        y = int(self.canvas_height * 0.92)
        return {
            0: (int(self.canvas_width * 0.27), y),
            1: (int(self.canvas_width * 0.41), y),
            2: (int(self.canvas_width * 0.56), y),
            3: (int(self.canvas_width * 0.70), y),
        }

    def _load_hand_slots(self, coords_file: str) -> dict[int, tuple[int, int]]:
        defaults = self._default_hand_slots()
        path = Path(coords_file)
        if not path.is_file():
            return defaults
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            slots_raw = data.get("hand_slots", {})
            parsed: dict[int, tuple[int, int]] = {}
            for k in ("0", "1", "2", "3"):
                if k not in slots_raw:
                    continue
                xy = slots_raw.get(k)
                if not isinstance(xy, list) or len(xy) != 2:
                    continue
                x, y = int(xy[0]), int(xy[1])
                parsed[int(k)] = (x, y)
            if len(parsed) == 4:
                return parsed
            return defaults
        except Exception:
            return defaults

    def _default_anchors(self) -> dict[str, tuple[int, int]]:
        return {
            "battle_button": (int(self.canvas_width * 0.50), int(self.canvas_height * 0.88)),
            "ok_button": (int(self.canvas_width * 0.50), int(self.canvas_height * 0.82)),
            "play_again_button": (int(self.canvas_width * 0.50), int(self.canvas_height * 0.84)),
            "left_bridge": (int(self.canvas_width * 0.36), int(self.canvas_height * 0.52)),
            "right_bridge": (int(self.canvas_width * 0.64), int(self.canvas_height * 0.52)),
        }

    def _load_anchors(self, coords_file: str) -> dict[str, tuple[int, int]]:
        defaults = self._default_anchors()
        path = Path(coords_file)
        if not path.is_file():
            return defaults
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            anchors_raw = data.get("anchors", {})
            parsed = dict(defaults)
            for key, xy in anchors_raw.items():
                if not isinstance(xy, list) or len(xy) != 2:
                    continue
                parsed[str(key)] = (int(xy[0]), int(xy[1]))
            return parsed
        except Exception:
            return defaults

    def tap_anchor(self, name: str) -> bool:
        xy = self.anchors.get(name)
        if not xy:
            return False
        return self.tap(xy[0], xy[1])

    def configure_touch_debug(self, show_touches: bool, pointer_location: bool) -> None:
        """
        Toggle Android touch debugging overlays visible inside MuMu.
        - show_touches: draw touch dot
        - pointer_location: draw pointer trail/coordinates
        """
        if self.dry_run:
            print(
                f"[ADB][DRY] touch_debug show_touches={int(show_touches)} pointer_location={int(pointer_location)}",
                flush=True,
            )
            return
        cmds = [
            ["adb", "shell", "settings", "put", "system", "show_touches", "1" if show_touches else "0"],
            ["adb", "shell", "settings", "put", "system", "pointer_location", "1" if pointer_location else "0"],
        ]
        for cmd in cmds:
            try:
                subprocess.run(cmd, check=False, capture_output=True, text=True)
            except Exception:
                pass

    def is_battle_view(self, frame: np.ndarray) -> bool:
        """Heuristic detector for in-match arena view."""
        h, w = frame.shape[:2]
        bottom = frame[int(h * 0.83) : h, :, :]
        if bottom.size == 0:
            return False
        gray = cv2.cvtColor(bottom, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(bottom, cv2.COLOR_BGR2HSV)
        mean = float(gray.mean())
        std = float(gray.std())
        sat = float(hsv[:, :, 1].mean())
        # Card bar is usually bright and textured during battle.
        return (mean > 35.0 and std > 11.0) or (sat > 45.0 and std > 8.0)

    def _fast_signals(self, frame: np.ndarray) -> dict[str, Any]:
        """
        Lightweight placeholder signal extraction.
        Replace with detector logic as your model matures.
        """
        h, w = frame.shape[:2]
        top_half = frame[: h // 2]
        gray = cv2.cvtColor(top_half, cv2.COLOR_BGR2GRAY)
        mean = float(gray.mean())
        left = float(gray[:, : w // 2].mean())
        right = float(gray[:, w // 2 :].mean())
        lane = "left" if left < right else "right"

        guessed = None
        if mean < 68:
            guessed = "log"

        return {
            "enemy_push_detected": mean < 88.0,
            "enemy_lane": lane,
            "enemy_card_guess": guessed,
        }

    def poll_state(self) -> ArenaState:
        t0 = time.perf_counter()
        frame = self.capture()
        t1 = time.perf_counter()
        return self.poll_state_from_frame(frame, cap_ms=(t1 - t0) * 1000.0)

    def poll_state_from_frame(self, frame: np.ndarray, cap_ms: float | None = None) -> ArenaState:
        t1 = time.perf_counter()
        signals = self._fast_signals(frame)
        t2 = time.perf_counter()

        guessed = signals.get("enemy_card_guess")
        if guessed:
            self.enemy_cards_seen.add(guessed)
            self.last_enemy_spend_t = time.perf_counter()
            if guessed == "log":
                self.enemy_log_available = False
                self.enemy_elixir_est = max(0, self.enemy_elixir_est - 2)

        regen = int((time.perf_counter() - self.last_enemy_spend_t) / 2.8)
        self.enemy_elixir_est = min(10, max(self.enemy_elixir_est, regen))

        cap_txt = f"{cap_ms:.1f}" if cap_ms is not None else "0.0"
        print(f"[LAT] capture={cap_txt}ms extract={(t2-t1)*1000:.1f}ms frame={self.frame_id}", flush=True)

        return ArenaState(
            ts=time.time(),
            frame_id=self.frame_id,
            enemy_cards_seen=sorted(self.enemy_cards_seen),
            enemy_elixir_est=self.enemy_elixir_est,
            enemy_log_available=self.enemy_log_available,
            teammate_context="teammate minions pressuring left",
            signals=signals,
        )
