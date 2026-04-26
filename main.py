from __future__ import annotations

import os
import queue
import threading
import time
from dataclasses import asdict

from brain import OllamaBrain
from context_bus import ContextBus
from debug_overlay import DebugOverlay, OverlayState
from model_perception import ModelPerception
from navigator import ClashNavigator
from vision import MuMuController

# Reuse existing project modules (no new Kokoro implementation).
from kokoro_speech import speak_kokoro
from stt_utils import hold_to_talk_to_text


FAST_MODEL = os.environ.get("FAST_MODEL", "gemma2:2b")
VISION_MODEL = os.environ.get("VISION_MODEL", "gemma4:e4b")
OLLAMA_TIMEOUT_S = float(os.environ.get("OLLAMA_TIMEOUT_S", "8"))
AUTO_NAVIGATE = os.environ.get("AUTO_NAVIGATE", "1") == "1"
USE_LLM_NAV = os.environ.get("USE_LLM_NAV", "1") == "1"
COMMENTARY_INTERVAL_S = float(os.environ.get("COMMENTARY_INTERVAL_S", "6"))
CONTEXT_REFRESH_S = float(os.environ.get("CONTEXT_REFRESH_S", "1.5"))
SHOW_TOUCHES = os.environ.get("SHOW_TOUCHES", "1") == "1"
POINTER_LOCATION = os.environ.get("POINTER_LOCATION", "0") == "1"
MODEL_OVERLAY = os.environ.get("MODEL_OVERLAY", "1") == "1"
MODEL_DETECTIONS = os.environ.get("MODEL_DETECTIONS", "1") == "1"
PERCEPTION_INTERVAL_S = float(os.environ.get("PERCEPTION_INTERVAL_S", "1.2"))
POLL_S = 0.04  # ~25Hz


class VoiceBus:
    """Non-blocking speech queue; drops stale lines under pressure."""

    def __init__(self) -> None:
        self.q: queue.Queue[str] = queue.Queue(maxsize=6)
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def say(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        if self.q.full():
            try:
                _ = self.q.get_nowait()
            except queue.Empty:
                pass
        self.q.put_nowait(text)

    def _worker(self) -> None:
        while not self.stop.is_set():
            try:
                line = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                speak_kokoro(line, voice="af_heart", lang_code="a", speed=1.12)
            except Exception as exc:
                print(f"[VOICE] failed: {exc}", flush=True)

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=1.0)


def stt_worker(out_q: queue.Queue[str], stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            text = hold_to_talk_to_text(
                hold_key="space",
                model_name="tiny.en",
                language="en",
                max_seconds=6.0,
            )
            text = (text or "").strip()
            if text:
                out_q.put(text)
        except Exception:
            time.sleep(0.15)


def main() -> int:
    print(f"[BOOT] FAST_MODEL={FAST_MODEL}", flush=True)
    print(f"[BOOT] VISION_MODEL={VISION_MODEL}", flush=True)
    print(f"[BOOT] DRY_RUN={os.environ.get('DRY_RUN', '0')}", flush=True)
    print(f"[BOOT] OLLAMA_TIMEOUT_S={OLLAMA_TIMEOUT_S}", flush=True)
    print(f"[BOOT] AUTO_NAVIGATE={int(AUTO_NAVIGATE)}", flush=True)
    print(f"[BOOT] USE_LLM_NAV={int(USE_LLM_NAV)}", flush=True)
    print(f"[BOOT] COMMENTARY_INTERVAL_S={COMMENTARY_INTERVAL_S}", flush=True)
    print(f"[BOOT] CONTEXT_REFRESH_S={CONTEXT_REFRESH_S}", flush=True)
    print(f"[BOOT] USE_ADB_SCREENSHOT={os.environ.get('USE_ADB_SCREENSHOT', '1')}", flush=True)
    print(f"[BOOT] SHOW_TOUCHES={int(SHOW_TOUCHES)} POINTER_LOCATION={int(POINTER_LOCATION)}", flush=True)
    print(f"[BOOT] MODEL_OVERLAY={int(MODEL_OVERLAY)}", flush=True)
    print(f"[BOOT] MODEL_DETECTIONS={int(MODEL_DETECTIONS)} PERCEPTION_INTERVAL_S={PERCEPTION_INTERVAL_S}", flush=True)

    controller = MuMuController(dry_run=os.environ.get("DRY_RUN", "0") == "1")
    navigator = ClashNavigator(controller)
    brain = OllamaBrain(model=FAST_MODEL, timeout_s=OLLAMA_TIMEOUT_S)
    context_bus = ContextBus(refresh_interval_s=CONTEXT_REFRESH_S)
    perception = ModelPerception() if MODEL_DETECTIONS else None
    voice = VoiceBus()
    overlay = DebugOverlay(enabled=MODEL_OVERLAY)
    controller.configure_touch_debug(SHOW_TOUCHES, POINTER_LOCATION)

    voice_cmd_q: queue.Queue[str] = queue.Queue(maxsize=8)
    stop_evt = threading.Event()
    stt_thread = threading.Thread(target=stt_worker, args=(voice_cmd_q, stop_evt), daemon=True)
    stt_thread.start()
    last_commentary_ts = 0.0
    last_battle_state = False
    last_perception_ts = 0.0
    model_detections: list[dict[str, object]] = []

    if AUTO_NAVIGATE:
        voice.say("Commander mode enabled. I will navigate and battle continuously.")

    try:
        while True:
            loop_t0 = time.perf_counter()

            newest_cmd = ""
            while True:
                try:
                    newest_cmd = voice_cmd_q.get_nowait()
                except queue.Empty:
                    break
            if newest_cmd:
                brain.set_voice_command(newest_cmd)
                voice.say(f"Heard: {newest_cmd}")

            frame = controller.capture()
            ctx = context_bus.snapshot()
            extra_ctx = {
                "focused_app": ctx.focused_app,
                "focused_activity": ctx.focused_activity,
                "ui_text": ctx.ui_text[:20],
                "ui_nodes": ctx.ui_nodes[:30],
            }
            in_battle = (
                navigator.step(frame, extra_context=extra_ctx)
                if AUTO_NAVIGATE
                else controller.is_battle_view(frame)
            )
            if perception is not None:
                nowp = time.time()
                if in_battle and (nowp - last_perception_ts >= PERCEPTION_INTERVAL_S):
                    model_detections = perception.infer(frame)
                    last_perception_ts = nowp
                elif not in_battle:
                    model_detections = []
            nav_cursor = navigator.last_cursor_xy if AUTO_NAVIGATE else None
            overlay.render(
                frame,
                overlay_state=OverlayState(
                    mode="battle" if in_battle else "nav",
                    screen_state=navigator.last_screen_state if AUTO_NAVIGATE else ("in_battle" if in_battle else "unknown"),
                    nav_action=navigator.last_applied_action if AUTO_NAVIGATE else "",
                    model_cursor_x=(nav_cursor[0] if nav_cursor else None),
                    model_cursor_y=(nav_cursor[1] if nav_cursor else None),
                    focused_app=ctx.focused_app,
                ),
                anchors=controller.anchors,
                ui_nodes=ctx.ui_nodes,
                model_detections=model_detections,
            )
            if not in_battle:
                last_battle_state = False
                elapsed = time.perf_counter() - loop_t0
                if elapsed < POLL_S:
                    time.sleep(POLL_S - elapsed)
                continue

            if not last_battle_state:
                voice.say("Battle confirmed. Running tactical mode.")
                last_battle_state = True

            state = asdict(controller.poll_state_from_frame(frame))
            plan = brain.decide_action(state, extra_context=extra_ctx)

            if plan.should_play:
                played = controller.play_card(plan.slot, plan.x, plan.y)
                if played:
                    line = brain.narrate_play(plan, state, extra_context=extra_ctx)
                    voice.say(line)
                    last_commentary_ts = time.time()
                    print(
                        f"[ACT] card={plan.card} slot={plan.slot} target=({plan.x},{plan.y}) "
                        f"where='{plan.where}' why='{plan.why}' conf={plan.confidence:.2f}",
                        flush=True,
                    )
                else:
                    print("[ACT] tap blocked by rate-limit or adb failure", flush=True)
            else:
                now = time.time()
                if now - last_commentary_ts >= COMMENTARY_INTERVAL_S:
                    voice.say(brain.narrate_status(state, extra_context=extra_ctx))
                    last_commentary_ts = now

            overlay.render(
                frame,
                overlay_state=OverlayState(
                    mode="battle",
                    screen_state="in_battle",
                    nav_action=f"play:{plan.card}" if plan.should_play else "hold",
                    model_cursor_x=(plan.x if plan.should_play else None),
                    model_cursor_y=(plan.y if plan.should_play else None),
                    battle_lane=state.get("signals", {}).get("enemy_lane", "unknown"),
                    enemy_elixir_est=state.get("enemy_elixir_est"),
                    focused_app=ctx.focused_app,
                ),
                anchors=controller.anchors,
                ui_nodes=ctx.ui_nodes,
                model_detections=model_detections,
            )

            elapsed = time.perf_counter() - loop_t0
            if elapsed < POLL_S:
                time.sleep(POLL_S - elapsed)

    except KeyboardInterrupt:
        print("\n[EXIT] stopping", flush=True)
    finally:
        stop_evt.set()
        voice.close()
        overlay.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
