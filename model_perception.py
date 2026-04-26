from __future__ import annotations

import base64
import json
import os
import re
import time
from typing import Any

import cv2
import numpy as np
import requests


class ModelPerception:
    """
    Ask vision model for overlay detections.
    Returns model-generated boxes for rendering.
    """

    def __init__(self) -> None:
        self.model = os.environ.get("VISION_MODEL", "gemma4:e4b")
        self.host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
        self.timeout_s = float(os.environ.get("PERCEPTION_TIMEOUT_S", "45"))
        self.max_tokens = int(os.environ.get("PERCEPTION_MAX_TOKENS", "220"))
        self.last_detections: list[dict[str, Any]] = []

    def _encode_frame(self, frame: np.ndarray) -> str:
        h, w = frame.shape[:2]
        max_side = max(h, w)
        if max_side > 720:
            scale = 720.0 / float(max_side)
            nw = max(1, int(w * scale))
            nh = max(1, int(h * scale))
            frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 55])
        if not ok:
            return ""
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    def _extract_json(self, text: str) -> dict[str, Any]:
        text = (text or "").strip()
        if not text:
            return {}
        try:
            return json.loads(text)
        except Exception:
            # Try to recover first JSON object in mixed output.
            m = re.search(r"\{.*\}", text, flags=re.S)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    return {}
            return {}

    def infer(self, frame: np.ndarray) -> list[dict[str, Any]]:
        img_b64 = self._encode_frame(frame)
        if not img_b64:
            return self.last_detections

        h, w = frame.shape[:2]
        prompt = (
            "You are VisionOverlay. Return strict JSON only.\n"
            "Find visible Clash Royale elements and units.\n"
            "Schema:\n"
            '{"detections":[{"label":"string","confidence":0.0,"team":"ally|enemy|neutral|unknown","bbox":[x1,y1,x2,y2]}]}\n'
            f"Image size is width={w}, height={h}. "
            "Use pixel coordinates in this image space. "
            "Include towers, obvious buttons/icons, and visible troops/minions if possible. "
            "Keep to at most 25 detections."
        )
        payload = {
            "model": self.model,
            "stream": False,
            "think": False,
            "messages": [{"role": "user", "content": prompt, "images": [img_b64]}],
            "options": {
                "temperature": 0.0,
                "num_predict": self.max_tokens,
            },
        }
        t0 = time.perf_counter()
        try:
            resp = requests.post(
                f"{self.host}/api/chat",
                json=payload,
                timeout=self.timeout_s,
            )
            resp.raise_for_status()
            content = resp.json().get("message", {}).get("content", "").strip()
            print(f"[PERCEPTION][RAW] {content}", flush=True)
            print(f"[PERCEPTION][LAT] {(time.perf_counter()-t0)*1000:.1f}ms", flush=True)
            obj = self._extract_json(content)
            raw = obj.get("detections", []) if isinstance(obj, dict) else []
            out: list[dict[str, Any]] = []
            for d in raw:
                if not isinstance(d, dict):
                    continue
                bbox = d.get("bbox")
                if not isinstance(bbox, list) or len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = [int(v) for v in bbox]
                x1 = max(0, min(w - 1, x1))
                y1 = max(0, min(h - 1, y1))
                x2 = max(0, min(w - 1, x2))
                y2 = max(0, min(h - 1, y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                out.append(
                    {
                        "label": str(d.get("label", "obj")),
                        "confidence": float(d.get("confidence", 0.5)),
                        "team": str(d.get("team", "unknown")).lower(),
                        "bbox": [x1, y1, x2, y2],
                    }
                )
            if out:
                self.last_detections = out[:25]
            return self.last_detections
        except Exception as exc:
            print(f"[PERCEPTION] fallback ({exc})", flush=True)
            return self.last_detections
