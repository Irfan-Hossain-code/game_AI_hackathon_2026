from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class OverlayState:
    mode: str = "nav"
    screen_state: str = "unknown"
    nav_action: str = ""
    model_cursor_x: int | None = None
    model_cursor_y: int | None = None
    battle_lane: str = "unknown"
    enemy_elixir_est: int | None = None
    focused_app: str = ""


class DebugOverlay:
    """
    Companion visual HUD (separate window) to inspect what the agent sees.
    """

    def __init__(self, window_name: str = "Agent Vision Overlay", enabled: bool = True) -> None:
        self.window_name = window_name
        self.enabled = enabled
        self._alive = enabled
        self._next_track_id = 1
        self._tracks: dict[int, dict[str, Any]] = {}
        self._max_missed = 10
        self._max_link_dist = 90.0

    def close(self) -> None:
        if not self._alive:
            return
        try:
            cv2.destroyWindow(self.window_name)
        except Exception:
            pass
        self._alive = False

    def _reset_tracks(self) -> None:
        self._tracks.clear()

    def _draw_towers(self, canvas: np.ndarray) -> None:
        h, w = canvas.shape[:2]
        # Approximate tower boxes for 1080x1920 style arena.
        boxes = {
            "enemy_left_tower": (int(w * 0.25), int(h * 0.22), int(w * 0.10), int(h * 0.08)),
            "enemy_right_tower": (int(w * 0.65), int(h * 0.22), int(w * 0.10), int(h * 0.08)),
            "enemy_king_tower": (int(w * 0.45), int(h * 0.14), int(w * 0.10), int(h * 0.08)),
            "ally_left_tower": (int(w * 0.25), int(h * 0.62), int(w * 0.10), int(h * 0.08)),
            "ally_right_tower": (int(w * 0.65), int(h * 0.62), int(w * 0.10), int(h * 0.08)),
            "ally_king_tower": (int(w * 0.45), int(h * 0.70), int(w * 0.10), int(h * 0.08)),
        }
        for name, (x, y, bw, bh) in boxes.items():
            cv2.rectangle(canvas, (x, y), (x + bw, y + bh), (255, 180, 0), 2)
            cv2.putText(
                canvas,
                name,
                (x, max(20, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 180, 0),
                1,
                cv2.LINE_AA,
            )

    def _detect_unit_candidates(self, canvas: np.ndarray) -> list[dict[str, Any]]:
        h, w = canvas.shape[:2]
        y0 = int(h * 0.18)
        y1 = int(h * 0.80)
        arena = canvas[y0:y1, :, :]
        hsv = cv2.cvtColor(arena, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        mask = cv2.inRange(sat, 60, 255) & cv2.inRange(val, 45, 255)
        mask = cv2.medianBlur(mask, 5)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections: list[dict[str, Any]] = []
        river_y = int((y0 + y1) * 0.5)
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 80 or area > 4500:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            x1, y1_abs = x + bw, y + bh + y0
            x0, y0_abs = x, y + y0
            cx = int((x0 + x1) * 0.5)
            cy = int((y0_abs + y1_abs) * 0.5)
            team = "enemy" if cy < river_y else "ally"
            detections.append(
                {
                    "bbox": (x0, y0_abs, x1, y1_abs),
                    "centroid": (cx, cy),
                    "area": float(area),
                    "team": team,
                }
            )
        return detections

    def _update_tracks(self, detections: list[dict[str, Any]]) -> None:
        track_ids = list(self._tracks.keys())
        used_det = set()
        used_tracks = set()

        # Greedy nearest-neighbor assignment
        pairs: list[tuple[float, int, int]] = []
        for tid in track_ids:
            tx, ty = self._tracks[tid]["centroid"]
            for di, det in enumerate(detections):
                dx, dy = det["centroid"]
                d = float(np.hypot(tx - dx, ty - dy))
                pairs.append((d, tid, di))
        pairs.sort(key=lambda t: t[0])

        for d, tid, di in pairs:
            if d > self._max_link_dist:
                continue
            if tid in used_tracks or di in used_det:
                continue
            used_tracks.add(tid)
            used_det.add(di)
            det = detections[di]
            tr = self._tracks[tid]
            tr["bbox"] = det["bbox"]
            tr["centroid"] = det["centroid"]
            tr["team"] = det["team"]
            tr["area"] = det["area"]
            tr["hits"] += 1
            tr["missed"] = 0
            tr["trail"].append(det["centroid"])
            if len(tr["trail"]) > 18:
                tr["trail"] = tr["trail"][-18:]

        # New tracks for unmatched detections
        for di, det in enumerate(detections):
            if di in used_det:
                continue
            tid = self._next_track_id
            self._next_track_id += 1
            self._tracks[tid] = {
                "bbox": det["bbox"],
                "centroid": det["centroid"],
                "team": det["team"],
                "area": det["area"],
                "hits": 1,
                "missed": 0,
                "trail": [det["centroid"]],
            }

        # Age unmatched tracks
        to_del = []
        for tid, tr in self._tracks.items():
            if tid not in used_tracks:
                tr["missed"] += 1
            if tr["missed"] > self._max_missed:
                to_del.append(tid)
        for tid in to_del:
            del self._tracks[tid]

    def _detections_from_model(self, model_detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for d in model_detections:
            bbox = d.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            x0, y0, x1, y1 = [int(v) for v in bbox]
            cx = int((x0 + x1) * 0.5)
            cy = int((y0 + y1) * 0.5)
            out.append(
                {
                    "bbox": (x0, y0, x1, y1),
                    "centroid": (cx, cy),
                    "area": float(max(1, (x1 - x0) * (y1 - y0))),
                    "team": str(d.get("team", "unknown")),
                    "label": str(d.get("label", "obj")),
                    "confidence": float(d.get("confidence", 0.5)),
                }
            )
        return out

    def _draw_tracks(self, canvas: np.ndarray) -> int:
        count = 0
        for tid, tr in self._tracks.items():
            x0, y0, x1, y1 = tr["bbox"]
            team = tr["team"]
            label = tr.get("label", "")
            if team == "enemy":
                color = (50, 120, 255)
            elif team == "ally":
                color = (80, 255, 80)
            else:
                color = (220, 220, 80)
            # confidence from hit/miss history
            conf_track = max(0.05, min(0.99, (tr["hits"] / max(1, tr["hits"] + tr["missed"]))))
            conf_model = float(tr.get("confidence", conf_track))
            tag = label[:14] if label else team[:1].upper()
            label_txt = f"{tag} #{tid} {conf_model:.2f}"

            cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)
            cv2.putText(
                canvas,
                label_txt,
                (x0, max(18, y0 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                color,
                1,
                cv2.LINE_AA,
            )
            trail = tr["trail"]
            for i in range(1, len(trail)):
                p0 = trail[i - 1]
                p1 = trail[i]
                alpha = i / max(1, len(trail))
                tcol = tuple(int(c * alpha) for c in color)
                cv2.line(canvas, p0, p1, tcol, 2)
            cx, cy = tr["centroid"]
            cv2.circle(canvas, (cx, cy), 3, color, -1)
            count += 1
        return count

    def _draw_anchors(self, canvas: np.ndarray, anchors: dict[str, tuple[int, int]]) -> None:
        for name, (x, y) in anchors.items():
            cv2.circle(canvas, (int(x), int(y)), 9, (0, 255, 0), 2)
            cv2.putText(
                canvas,
                name,
                (int(x) + 10, int(y) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

    def _draw_ui_nodes(self, canvas: np.ndarray, ui_nodes: list[dict[str, Any]] | None) -> int:
        if not ui_nodes:
            return 0
        drawn = 0
        for n in ui_nodes[:20]:
            bounds = n.get("bounds")
            txt = str(n.get("text", "")).strip()
            if not isinstance(bounds, list) or len(bounds) != 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in bounds]
            if x2 <= x1 or y2 <= y1:
                continue
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (170, 120, 255), 1)
            if txt:
                cv2.putText(
                    canvas,
                    txt[:22],
                    (x1, max(14, y1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.36,
                    (170, 120, 255),
                    1,
                    cv2.LINE_AA,
                )
            drawn += 1
        return drawn

    def _draw_cursor(self, canvas: np.ndarray, x: int | None, y: int | None, label: str) -> None:
        if x is None or y is None:
            return
        x, y = int(x), int(y)
        cv2.drawMarker(canvas, (x, y), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=24, thickness=2)
        cv2.circle(canvas, (x, y), 14, (0, 0, 255), 2)
        cv2.putText(
            canvas,
            f"model_cursor: {label}",
            (x + 12, y + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    def render(
        self,
        frame: np.ndarray,
        *,
        overlay_state: OverlayState,
        anchors: dict[str, tuple[int, int]],
        ui_nodes: list[dict[str, Any]] | None = None,
        model_detections: list[dict[str, Any]] | None = None,
    ) -> None:
        if not self.enabled or not self._alive:
            return
        try:
            canvas = frame.copy()
            in_battle = overlay_state.mode == "battle" or overlay_state.screen_state == "in_battle"
            if not in_battle:
                self._reset_tracks()

            if in_battle:
                self._draw_towers(canvas)
            self._draw_anchors(canvas, anchors)
            ui_count = self._draw_ui_nodes(canvas, ui_nodes)
            # If model detections are provided, use only them (no heuristic fallback).
            detections = self._detections_from_model(model_detections or [])
            self._update_tracks(detections)
            track_count = self._draw_tracks(canvas)
            self._draw_cursor(
                canvas,
                overlay_state.model_cursor_x,
                overlay_state.model_cursor_y,
                overlay_state.nav_action or overlay_state.mode,
            )

            lines = [
                f"mode={overlay_state.mode} screen={overlay_state.screen_state}",
                f"focused_app={overlay_state.focused_app}",
                f"nav_action={overlay_state.nav_action}",
                f"lane={overlay_state.battle_lane} enemy_elixir={overlay_state.enemy_elixir_est}",
                f"ui_nodes={ui_count} detections={len(detections)} tracks={track_count}",
            ]
            y = 24
            for line in lines:
                cv2.putText(canvas, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(canvas, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 20), 1, cv2.LINE_AA)
                y += 24

            cv2.imshow(self.window_name, canvas)
            cv2.waitKey(1)
        except Exception as exc:
            print(f"[OVERLAY] disabled ({exc})", flush=True)
            self._alive = False
