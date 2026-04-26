from __future__ import annotations

import base64
import json
import os
import re
import time
from collections import deque
from typing import Any

import cv2
import numpy as np
import requests

from screen_classifier import ScreenClassifier
from vision import MuMuController


class ClashNavigator:
    """
    Minimal commander-style navigator:
    - launches Clash Royale package
    - taps battle/home buttons when not in-match
    - exits once battle view is detected
    """

    def __init__(self, controller: MuMuController) -> None:
        self.controller = controller
        self.package_name = os.environ.get("CR_PACKAGE", "com.supercell.clashroyale")
        self.nav_model = os.environ.get(
            "VISION_MODEL",
            os.environ.get("NAV_MODEL", "gemma4:e4b"),
        )
        self.nav_timeout_s = float(os.environ.get("NAV_TIMEOUT_S", "60"))
        self.nav_max_tokens = int(os.environ.get("NAV_MAX_TOKENS", "80"))
        self.use_llm_nav = os.environ.get("USE_LLM_NAV", "1") == "1"
        self.ollama_host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
        self.classifier = ScreenClassifier()
        self.nav_history: deque[dict[str, Any]] = deque(maxlen=16)
        self.stuck_counter = 0
        self.last_state_name = ""
        self.recovery_idx = 0
        self.no_progress_count = 0
        self.prev_signature = ""
        self.last_applied_action = ""
        self.same_action_count = 0
        self.last_screen_state = "unknown"
        self.last_cursor_xy: tuple[int, int] | None = None
        self.nav_role_prompt = (
            "You are NavMind, a Clash Royale screen navigator. "
            "Given a screenshot, classify screen and choose one safe UI action. "
            "Return strict JSON only."
        )
        self.launched = False
        self.last_action_t = 0.0

    def launch_game(self) -> bool:
        cmd = f"adb shell monkey -p {self.package_name} 1"
        if self.controller.dry_run:
            print(f"[ADB][DRY] {cmd}", flush=True)
            return True
        return os.system(cmd) == 0

    def _try_menu_actions(self) -> None:
        for key in ("ok_button", "battle_button", "play_again_button"):
            self.controller.tap_anchor(key)
            time.sleep(0.15)

    def _action_from_state(self, state_name: str) -> str:
        if state_name == "outside_game":
            return "launch_game"
        if state_name == "popup":
            return "tap_ok_button"
        if state_name == "menu":
            return "tap_battle_button"
        return "wait"

    def _is_outside_game(self, extra_context: dict[str, Any] | None) -> bool:
        if not extra_context:
            return False
        focused_app = str(extra_context.get("focused_app", "") or "").strip().lower()
        focused_activity = str(extra_context.get("focused_activity", "") or "").strip().lower()
        ui_text = [str(t).lower() for t in (extra_context.get("ui_text") or [])]
        pkg = self.package_name.lower()

        if focused_app and pkg not in focused_app:
            return True
        if focused_activity and pkg not in focused_activity:
            # Some ROMs hide package in one field; fall through if ui strongly suggests in-game.
            if not any("battle" in t or "clash royale" in t for t in ui_text):
                return True
        # Launcher/home hints.
        launcher_hints = ("play store", "settings", "applications")
        if any(any(h in t for h in launcher_hints) for t in ui_text) and not any(
            "battle" in t or "clash royale" in t for t in ui_text
        ):
            return True
        return False

    def _tap_ui_text(self, extra_context: dict[str, Any] | None, keywords: list[str]) -> bool:
        if not extra_context:
            return False
        nodes = extra_context.get("ui_nodes") or []
        for n in nodes:
            text = str(n.get("text", "")).strip().lower()
            if not text:
                continue
            if not any(k.lower() in text for k in keywords):
                continue
            center = n.get("center")
            if isinstance(center, list) and len(center) == 2:
                self.last_cursor_xy = (int(center[0]), int(center[1]))
                return self.controller.tap(int(center[0]), int(center[1]))
        return False

    def _frame_signature(self, frame) -> str:
        """
        Tiny signature to detect whether UI changed after taps.
        """
        h, w = frame.shape[:2]
        if h == 0 or w == 0:
            return "empty"
        roi = frame[int(h * 0.20) : int(h * 0.92), int(w * 0.12) : int(w * 0.88)]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
        mean = float(np.mean(small))
        std = float(np.std(small))
        bins = np.histogram(small, bins=8, range=(0, 256))[0]
        return f"{int(mean)}-{int(std)}-" + ",".join(str(int(x)) for x in bins.tolist())

    def _encode_frame(self, frame) -> str:
        # Keep screenshot payload small for faster multimodal inference.
        h, w = frame.shape[:2]
        max_side = max(h, w)
        if max_side > 640:
            scale = 640.0 / float(max_side)
            nw = max(1, int(w * scale))
            nh = max(1, int(h * scale))
            frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 45])
        if not ok:
            return ""
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    def _llm_decide_action(
        self,
        frame,
        state_name: str,
        metrics: dict[str, Any],
        extra_context: dict[str, Any] | None = None,
    ) -> str:
        if not self.use_llm_nav:
            return self._action_from_state(state_name)
        extra_context = extra_context or {}
        img_b64 = self._encode_frame(frame)
        if not img_b64:
            return self._action_from_state(state_name)
        anchors = sorted(self.controller.anchors.keys())
        prompt = (
            f"{self.nav_role_prompt}\n"
            "Return strict JSON only: "
            '{"state":"menu|popup|in_battle|unknown","action":"tap_battle_button|tap_ok_button|tap_play_again_button|tap_anchor|tap_xy|back|launch_game|wait","anchor":"string","x":0,"y":0}\n'
            f"Heuristic_state={state_name}, metrics={metrics}, stuck_counter={self.stuck_counter}, "
            f"known_anchors={anchors}, recent_nav_history={list(self.nav_history)}.\n"
            f"extra_context={json.dumps(extra_context, ensure_ascii=True)}.\n"
            "If stuck_counter>=4, try a recovery action (back, launch_game, or different anchor). "
            "Keep output under 20 tokens."
        )
        payload = {
            "model": self.nav_model,
            "stream": False,
            "think": False,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [img_b64],
                }
            ],
            "options": {
                "temperature": 0.0,
                "num_predict": self.nav_max_tokens,
            },
        }
        try:
            resp = requests.post(
                f"{self.ollama_host.rstrip('/')}/api/chat",
                json=payload,
                timeout=self.nav_timeout_s,
            )
            resp.raise_for_status()
            content = (
                resp.json()
                .get("message", {})
                .get("content", "")
                .strip()
            )
            print(f"[NAV][RAW] {content}", flush=True)
            obj: dict[str, Any] = {}
            action = ""
            if content:
                try:
                    obj = json.loads(content)
                except Exception:
                    # Recover key fields from truncated JSON responses.
                    m_action = re.search(r'"action"\s*:\s*"([^"]+)"', content)
                    if m_action:
                        obj["action"] = m_action.group(1)
                    m_anchor = re.search(r'"anchor"\s*:\s*"([^"]+)"', content)
                    if m_anchor:
                        obj["anchor"] = m_anchor.group(1)
                    m_x = re.search(r'"x"\s*:\s*(-?\d+)', content)
                    m_y = re.search(r'"y"\s*:\s*(-?\d+)', content)
                    if m_x:
                        obj["x"] = int(m_x.group(1))
                    if m_y:
                        obj["y"] = int(m_y.group(1))
            action = str(obj.get("action", "")).strip()
            if action in {
                "tap_battle_button",
                "tap_ok_button",
                "tap_play_again_button",
                "tap_anchor",
                "tap_xy",
                "back",
                "launch_game",
                "wait",
            }:
                return json.dumps(obj)
        except Exception as exc:
            print(f"[NAV] llm-nav fallback ({exc})", flush=True)
        return self._action_from_state(state_name)

    def _apply_action(self, action_payload: str, unknown_cycle_idx: int) -> tuple[int, str]:
        action = action_payload
        anchor = ""
        x = None
        y = None
        if action_payload.startswith("{"):
            try:
                obj = json.loads(action_payload)
                action = str(obj.get("action", "wait"))
                anchor = str(obj.get("anchor", "")).strip()
                if "x" in obj and "y" in obj:
                    x = int(obj.get("x"))
                    y = int(obj.get("y"))
            except Exception:
                action = "wait"

        if action == "tap_battle_button":
            xy = self.controller.anchors.get("battle_button")
            self.last_cursor_xy = xy
            ok = self.controller.tap_anchor("battle_button")
            return unknown_cycle_idx, f"{action}:{'ok' if ok else 'fail'}"
        elif action == "tap_ok_button":
            xy = self.controller.anchors.get("ok_button")
            self.last_cursor_xy = xy
            ok = self.controller.tap_anchor("ok_button")
            return unknown_cycle_idx, f"{action}:{'ok' if ok else 'fail'}"
        elif action == "tap_play_again_button":
            xy = self.controller.anchors.get("play_again_button")
            self.last_cursor_xy = xy
            ok = self.controller.tap_anchor("play_again_button")
            return unknown_cycle_idx, f"{action}:{'ok' if ok else 'fail'}"
        elif action == "tap_anchor" and anchor:
            self.last_cursor_xy = self.controller.anchors.get(anchor)
            if self.controller.tap_anchor(anchor):
                return unknown_cycle_idx, f"{action}:{anchor}"
            return unknown_cycle_idx, f"{action}:missing_{anchor}"
        elif action == "tap_xy" and x is not None and y is not None:
            self.last_cursor_xy = (x, y)
            self.controller.tap(x, y)
            return unknown_cycle_idx, f"{action}:{x},{y}"
        elif action == "back":
            self.last_cursor_xy = None
            self.controller.back()
            return unknown_cycle_idx, action
        elif action == "launch_game":
            self.last_cursor_xy = None
            self.launch_game()
            return unknown_cycle_idx, action
        elif action == "wait":
            return unknown_cycle_idx, action
        else:
            cycle = ("ok_button", "battle_button", "play_again_button")
            key = cycle[unknown_cycle_idx % len(cycle)]
            ok = self.controller.tap_anchor(key)
            unknown_cycle_idx += 1
            return unknown_cycle_idx, f"fallback:{key}:{'ok' if ok else 'fail'}"

    def _deterministic_recovery(self) -> str:
        sequence = (
            "back",
            "tap_ok_button",
            "tap_play_again_button",
            "tap_battle_button",
            "tap_battle_button_jitter_left",
            "tap_battle_button_jitter_right",
        )
        action = sequence[self.recovery_idx % len(sequence)]
        self.recovery_idx += 1

        if action == "back":
            ok = self.controller.back()
            return f"{action}:{'ok' if ok else 'fail'}"
        if action == "tap_ok_button":
            ok = self.controller.tap_anchor("ok_button")
            return f"{action}:{'ok' if ok else 'fail'}"
        if action == "tap_play_again_button":
            ok = self.controller.tap_anchor("play_again_button")
            return f"{action}:{'ok' if ok else 'fail'}"
        if action == "tap_battle_button":
            ok = self.controller.tap_anchor("battle_button")
            return f"{action}:{'ok' if ok else 'fail'}"
        if action == "tap_battle_button_jitter_left":
            x, y = self.controller.anchors.get("battle_button", (540, 1690))
            ok = self.controller.tap(x - 40, y)
            return f"{action}:{'ok' if ok else 'fail'}"
        if action == "tap_battle_button_jitter_right":
            x, y = self.controller.anchors.get("battle_button", (540, 1690))
            ok = self.controller.tap(x + 40, y)
            return f"{action}:{'ok' if ok else 'fail'}"
        return "wait"

    def _deterministic_menu_action(self) -> str:
        """
        Fast, reliable menu policy before asking LLM:
        - try battle button
        - then slight x jitter left/right
        - occasionally clear popups/back
        """
        c = self.stuck_counter
        if c % 9 == 8:
            return "back"
        if c % 7 == 6:
            return "tap_ok_button"
        if c % 5 == 4:
            return "tap_play_again_button"
        if c % 3 == 1:
            return "tap_battle_button_jitter_left"
        if c % 3 == 2:
            return "tap_battle_button_jitter_right"
        return "tap_battle_button"

    def _apply_simple_action(self, action: str) -> str:
        if action == "back":
            ok = self.controller.back()
            return f"{action}:{'ok' if ok else 'fail'}"
        if action == "tap_ok_button":
            ok = self.controller.tap_anchor("ok_button")
            return f"{action}:{'ok' if ok else 'fail'}"
        if action == "tap_play_again_button":
            ok = self.controller.tap_anchor("play_again_button")
            return f"{action}:{'ok' if ok else 'fail'}"
        if action == "tap_battle_button":
            ok = self.controller.tap_anchor("battle_button")
            return f"{action}:{'ok' if ok else 'fail'}"
        if action == "tap_battle_button_jitter_left":
            x, y = self.controller.anchors.get("battle_button", (540, 1690))
            ok = self.controller.tap(x - 40, y)
            return f"{action}:{'ok' if ok else 'fail'}"
        if action == "tap_battle_button_jitter_right":
            x, y = self.controller.anchors.get("battle_button", (540, 1690))
            ok = self.controller.tap(x + 40, y)
            return f"{action}:{'ok' if ok else 'fail'}"
        return "wait"

    def step(
        self,
        frame,
        min_action_interval_s: float = 1.8,
        extra_context: dict[str, Any] | None = None,
    ) -> bool:
        """
        One commander tick.
        Returns True when battle view is detected, else False.
        """
        screen = self.classifier.classify(frame)
        if self._is_outside_game(extra_context):
            screen.name = "outside_game"
            screen.confidence = 0.95
            screen.metrics = {
                **screen.metrics,
                "focused_app": (extra_context or {}).get("focused_app", ""),
                "focused_activity": (extra_context or {}).get("focused_activity", ""),
            }
        sig = self._frame_signature(frame)
        if self.prev_signature and sig == self.prev_signature:
            self.no_progress_count += 1
        else:
            self.no_progress_count = 0
        self.prev_signature = sig
        if screen.name == "in_battle" or self.controller.is_battle_view(frame):
            if self.stuck_counter:
                self.stuck_counter = 0
            self.no_progress_count = 0
            self.last_applied_action = ""
            self.same_action_count = 0
            self.last_screen_state = "in_battle"
            return True

        if screen.name == self.last_state_name:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
            self.last_state_name = screen.name

        now = time.perf_counter()
        if not self.launched:
            self.launched = self.launch_game()
            self.last_action_t = now
            print(f"[NAV] launch sent={self.launched}", flush=True)
            return False

        if now - self.last_action_t < min_action_interval_s:
            return False

        applied = "wait"
        if screen.name == "outside_game":
            # First try semantic icon/button tap from UI nodes, then launch.
            tapped_icon = self._tap_ui_text(extra_context, ["clash royale", "play"])
            if tapped_icon:
                applied = "tap_ui_text:ok"
            else:
                ok = self.launch_game()
                applied = f"launch_game:{'ok' if ok else 'fail'}"
            self.last_action_t = now
            self.last_screen_state = "outside_game"
            self.nav_history.append(
                {
                    "state": "outside_game",
                    "stuck": self.stuck_counter,
                    "no_progress": self.no_progress_count,
                    "applied": applied,
                    "focused_app": (extra_context or {}).get("focused_app", ""),
                }
            )
            print(
                f"[NAV] state=outside_game conf={screen.confidence:.2f} action={applied} stuck={self.stuck_counter} no_progress={self.no_progress_count} model={self.nav_model}",
                flush=True,
            )
            return False

        # Deterministic first for menu; use LLM for popup/unknown or deep-stuck.
        if screen.name == "menu" and self.no_progress_count < 6:
            applied = self._apply_simple_action(self._deterministic_menu_action())
        elif screen.name == "menu" and self.no_progress_count >= 6:
            applied = f"recovery:{self._deterministic_recovery()}"
        elif screen.name in {"popup", "unknown"} and self.use_llm_nav:
            action_payload = self._llm_decide_action(frame, screen.name, screen.metrics, extra_context=extra_context)
            unknown_cycle_idx = len(self.nav_history)
            unknown_cycle_idx, applied = self._apply_action(action_payload, unknown_cycle_idx)
            _ = unknown_cycle_idx
        else:
            applied = self._apply_simple_action(self._action_from_state(screen.name))

        # Anti-repeat: if same action repeats too much, force a different recovery.
        if applied == self.last_applied_action:
            self.same_action_count += 1
        else:
            self.same_action_count = 0
            self.last_applied_action = applied
        if self.same_action_count >= 3:
            applied = f"anti_repeat:{self._deterministic_recovery()}"
            self.same_action_count = 0
            self.last_applied_action = applied

        self.nav_history.append(
            {
                "state": screen.name,
                "stuck": self.stuck_counter,
                "no_progress": self.no_progress_count,
                "applied": applied,
            }
        )
        self.last_action_t = now
        self.last_screen_state = screen.name
        print(
            f"[NAV] state={screen.name} conf={screen.confidence:.2f} action={applied} stuck={self.stuck_counter} no_progress={self.no_progress_count} model={self.nav_model}",
            flush=True,
        )
        return False

    def ensure_battle(self, max_wait_s: float = 45.0) -> bool:
        t0 = time.perf_counter()
        launched = False
        last_action_t = 0.0
        unknown_cycle_idx = 0

        while time.perf_counter() - t0 < max_wait_s:
            try:
                frame = self.controller.capture()
            except Exception as exc:
                print(f"[NAV] capture error: {exc}", flush=True)
                time.sleep(0.5)
                continue

            screen = self.classifier.classify(frame)
            if screen.name == "in_battle" or self.controller.is_battle_view(frame):
                print("[NAV] battle view detected", flush=True)
                return True
            if screen.name == self.last_state_name:
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0
                self.last_state_name = screen.name

            now = time.perf_counter()
            if not launched:
                launched = self.launch_game()
                print(f"[NAV] launch sent={launched}", flush=True)
                last_action_t = now
            elif now - last_action_t > 2.5:
                action_payload = self._llm_decide_action(frame, screen.name, screen.metrics)
                unknown_cycle_idx, applied = self._apply_action(action_payload, unknown_cycle_idx)
                self.nav_history.append(
                    {
                        "state": screen.name,
                        "stuck": self.stuck_counter,
                        "applied": applied,
                    }
                )
                print(
                    f"[NAV] state={screen.name} conf={screen.confidence:.2f} action={applied} stuck={self.stuck_counter} model={self.nav_model}",
                    flush=True,
                )
                last_action_t = now

            time.sleep(0.35)

        print("[NAV] timeout waiting for battle view", flush=True)
        return False
