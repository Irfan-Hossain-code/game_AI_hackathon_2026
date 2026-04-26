from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class ScreenState:
    name: str
    confidence: float
    metrics: dict[str, Any]


class ScreenClassifier:
    """
    Lightweight screen-state classifier.
    Keeps navigation deterministic and low-latency.
    """

    def classify(self, frame: np.ndarray) -> ScreenState:
        h, w = frame.shape[:2]
        if h == 0 or w == 0:
            return ScreenState("unknown", 0.0, {})

        bottom = frame[int(h * 0.82) : h, :, :]
        center = frame[int(h * 0.30) : int(h * 0.74), int(w * 0.18) : int(w * 0.82), :]
        top = frame[: int(h * 0.20), :, :]

        b_gray = cv2.cvtColor(bottom, cv2.COLOR_BGR2GRAY)
        b_hsv = cv2.cvtColor(bottom, cv2.COLOR_BGR2HSV)
        c_gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)
        t_gray = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY)

        bottom_mean = float(np.mean(b_gray))
        bottom_std = float(np.std(b_gray))
        bottom_sat = float(np.mean(b_hsv[:, :, 1]))
        center_mean = float(np.mean(c_gray))
        center_std = float(np.std(c_gray))
        top_mean = float(np.mean(t_gray))

        # Arena card bar usually has texture and/or saturation in lower UI.
        if (bottom_mean > 35.0 and bottom_std > 11.0) or (bottom_sat > 45.0 and bottom_std > 8.0):
            return ScreenState(
                "in_battle",
                0.86,
                {
                    "bottom_mean": bottom_mean,
                    "bottom_std": bottom_std,
                    "bottom_sat": bottom_sat,
                },
            )

        # Popups often brighten center and flatten texture.
        if center_mean > top_mean + 12.0 and center_std < 28.0:
            return ScreenState(
                "popup",
                0.68,
                {
                    "center_mean": center_mean,
                    "center_std": center_std,
                    "top_mean": top_mean,
                },
            )

        # Menu-ish fallback: no battle bar + more uniform bottom.
        if bottom_std < 14.0:
            return ScreenState(
                "menu",
                0.62,
                {
                    "bottom_mean": bottom_mean,
                    "bottom_std": bottom_std,
                    "bottom_sat": bottom_sat,
                },
            )

        return ScreenState(
            "unknown",
            0.35,
            {
                "bottom_mean": bottom_mean,
                "bottom_std": bottom_std,
                "bottom_sat": bottom_sat,
                "center_mean": center_mean,
                "center_std": center_std,
            },
        )
