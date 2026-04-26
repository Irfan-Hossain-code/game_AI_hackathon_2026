#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import subprocess
import time

import cv2
import numpy as np


def adb_screencap(timeout_s: float = 1.0) -> np.ndarray | None:
    """Capture current Android framebuffer through ADB."""
    try:
        proc = subprocess.run(
            ["adb", "exec-out", "screencap", "-p"],
            capture_output=True,
            check=False,
            timeout=timeout_s,
        )
        if proc.returncode != 0 or not proc.stdout:
            return None
        buf = np.frombuffer(proc.stdout, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return frame
    except Exception:
        return None


def mss_screencap(mss_module, sct, monitor_index: int = 1) -> np.ndarray | None:
    """Capture desktop monitor via MSS (very fast/live)."""
    try:
        monitors = sct.monitors
        idx = min(max(1, int(monitor_index)), len(monitors) - 1 if len(monitors) > 1 else 1)
        mon = monitors[idx]
        shot = sct.grab(mon)
        frame = np.array(shot, dtype=np.uint8)
        # BGRA -> BGR
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    except Exception:
        return None


def draw_hud(frame: np.ndarray, lines: list[str]) -> None:
    y = 24
    for line in lines:
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
        y += 22


def nms_xyxy(boxes: list[list[int]], scores: list[float], iou_thresh: float) -> list[int]:
    if not boxes:
        return []
    bxywh: list[list[int]] = []
    for x1, y1, x2, y2 in boxes:
        bxywh.append([x1, y1, max(1, x2 - x1), max(1, y2 - y1)])
    idxs = cv2.dnn.NMSBoxes(bxywh, scores, score_threshold=0.01, nms_threshold=iou_thresh)
    if idxs is None or len(idxs) == 0:
        return []
    out: list[int] = []
    for v in idxs:
        if isinstance(v, (list, tuple, np.ndarray)):
            out.append(int(v[0]))
        else:
            out.append(int(v))
    return out


def valid_ocr_text(s: str) -> bool:
    s = s.strip()
    if len(s) < 2:
        return False
    return bool(re.search(r"[A-Za-z0-9]", s))


def main() -> int:
    ap = argparse.ArgumentParser(description="OpenCV-only live overlay from emulator screen.")
    ap.add_argument("--window", default="OpenCV Live Analyzer")
    ap.add_argument("--max-fps", type=float, default=12.0, help="Overlay refresh cap (default: 12)")
    ap.add_argument("--ocr", action="store_true", help="Enable OCR overlay for detected words")
    ap.add_argument("--ocr-min-conf", type=int, default=45, help="Min OCR confidence (0-100)")
    ap.add_argument("--show-mser", action="store_true", help="Show raw MSER text-like boxes")
    ap.add_argument("--max-region-boxes", type=int, default=40)
    ap.add_argument("--max-ocr-boxes", type=int, default=35)
    ap.add_argument("--nms-iou", type=float, default=0.35)
    ap.add_argument("--use-cr-model", action="store_true", help="Use local Clash Royale detector model")
    ap.add_argument("--cr-model-path", default=os.environ.get("CR_MODEL_PATH", "models/clash_royale.pt"))
    ap.add_argument("--clean-ui", action="store_true", help="Minimal, stable UI-only overlay preset")
    ap.add_argument("--ocr-interval-s", type=float, default=0.55, help="Seconds between OCR refreshes")
    ap.add_argument("--model-interval-s", type=float, default=0.8, help="Seconds between model refreshes")
    ap.add_argument(
        "--capture-backend",
        choices=["auto", "mss", "adb"],
        default="auto",
        help="Frame capture backend. auto prefers mss for live refresh.",
    )
    ap.add_argument("--monitor-index", type=int, default=1, help="Monitor index for mss capture")
    ap.add_argument("--adb-timeout-s", type=float, default=1.0, help="ADB screencap timeout seconds")
    args = ap.parse_args()

    min_dt = 1.0 / max(1.0, args.max_fps)
    prev_gray: np.ndarray | None = None
    prev_t = time.perf_counter()
    ocr_enabled = False
    pytesseract = None

    if args.ocr:
        try:
            import pytesseract as _pytesseract  # type: ignore

            pytesseract = _pytesseract
            ocr_enabled = True
        except Exception:
            ocr_enabled = False

    yolo = None
    if args.use_cr_model and os.path.exists(args.cr_model_path):
        try:
            from ultralytics import YOLO  # type: ignore

            yolo = YOLO(args.cr_model_path)
        except Exception:
            yolo = None

    mss_module = None
    sct = None
    if args.capture_backend in ("auto", "mss"):
        try:
            import mss as _mss  # type: ignore

            mss_module = _mss
            sct = _mss.mss()
        except Exception:
            mss_module = None
            sct = None

    last_ocr_t = 0.0
    last_ocr_boxes: list[list[int]] = []
    last_ocr_labels: list[str] = []
    last_model_t = 0.0
    last_model_boxes: list[tuple[int, int, int, int]] = []
    last_model_labels: list[str] = []

    while True:
        t0 = time.perf_counter()
        used_backend = "none"
        frame = None
        if args.capture_backend == "adb":
            frame = adb_screencap(timeout_s=args.adb_timeout_s)
            used_backend = "adb"
        elif args.capture_backend == "mss":
            if mss_module is not None and sct is not None:
                frame = mss_screencap(mss_module, sct, args.monitor_index)
                used_backend = "mss"
        else:
            if mss_module is not None and sct is not None:
                frame = mss_screencap(mss_module, sct, args.monitor_index)
                used_backend = "mss"
            if frame is None:
                frame = adb_screencap(timeout_s=args.adb_timeout_s)
                used_backend = "adb"

        if frame is None:
            blank = np.zeros((480, 854, 3), dtype=np.uint8)
            draw_hud(
                blank,
                [
                    "Capture failed",
                    f"backend={args.capture_backend} (auto picks mss first)",
                    "Check: adb devices OR install mss",
                    "Press q to quit",
                ],
            )
            cv2.imshow(args.window, blank)
            if (cv2.waitKey(30) & 0xFF) == ord("q"):
                break
            continue

        h, w = frame.shape[:2]
        view = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 1) Generic "object-like" saturated regions
        sat_mask = cv2.inRange(hsv[:, :, 1], 80, 255)
        val_mask = cv2.inRange(hsv[:, :, 2], 50, 255)
        obj_mask = cv2.bitwise_and(sat_mask, val_mask)
        obj_mask = cv2.morphologyEx(obj_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        obj_mask = cv2.morphologyEx(obj_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        cnts, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        region_boxes: list[list[int]] = []
        region_scores: list[float] = []
        if args.clean_ui:
            # Focus only on stable UI zones when clean mode is requested.
            ui_rois = [
                (0, 0, w, int(h * 0.18)),  # top bar
                (int(w * 0.12), int(h * 0.74), int(w * 0.76), int(h * 0.16)),  # cards / tabs / CTA zone
                (int(w * 0.22), int(h * 0.77), int(w * 0.56), int(h * 0.12)),  # main CTA (battle-like) zone
            ]
        else:
            ui_rois = [(0, 0, w, h)]
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 250 or area > 22000:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            if bw < 12 or bh < 12:
                continue
            in_roi = False
            for rx, ry, rw, rh in ui_rois:
                if x >= rx and y >= ry and (x + bw) <= (rx + rw) and (y + bh) <= (ry + rh):
                    in_roi = True
                    break
            if not in_roi:
                continue
            region_boxes.append([x, y, x + bw, y + bh])
            region_scores.append(float(area))
        obj_count = 0
        keep_regions = nms_xyxy(region_boxes, region_scores, args.nms_iou)
        max_region_boxes = min(args.max_region_boxes, 12) if args.clean_ui else args.max_region_boxes
        for i in keep_regions[:max_region_boxes]:
            x1, y1, x2, y2 = region_boxes[i]
            cv2.rectangle(view, (x1, y1), (x2, y2), (0, 220, 255), 1)
            obj_count += 1

        if args.clean_ui:
            # Stable guide boxes for key menu UI zones.
            top = (0, 0, w, int(h * 0.18))
            cards = (int(w * 0.12), int(h * 0.74), int(w * 0.76), int(h * 0.16))
            cta = (int(w * 0.22), int(h * 0.77), int(w * 0.56), int(h * 0.12))
            for label, (rx, ry, rw, rh), color in [
                ("top_bar", top, (255, 180, 0)),
                ("cards_area", cards, (255, 160, 0)),
                ("primary_cta", cta, (255, 140, 0)),
            ]:
                cv2.rectangle(view, (rx, ry), (rx + rw, ry + rh), color, 2)
                cv2.putText(
                    view,
                    label,
                    (rx + 6, max(16, ry + 18)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        # 2) Optional raw text-like regions with MSER
        text_like = 0
        if args.show_mser:
            mser = cv2.MSER_create()
            mser.setMinArea(80)
            mser.setMaxArea(3500)
            regions, _ = mser.detectRegions(gray)
            mser_boxes: list[list[int]] = []
            mser_scores: list[float] = []
            for pts in regions[:400]:
                x, y, bw, bh = cv2.boundingRect(pts.reshape(-1, 1, 2))
                if bw < 10 or bh < 10 or bw > 220 or bh > 80:
                    continue
                ar = bw / max(1.0, bh)
                if ar < 0.4 or ar > 10.0:
                    continue
                mser_boxes.append([x, y, x + bw, y + bh])
                mser_scores.append(float(bw * bh))
            mkeep = nms_xyxy(mser_boxes, mser_scores, 0.30)
            for i in mkeep[:25]:
                x1, y1, x2, y2 = mser_boxes[i]
                cv2.rectangle(view, (x1, y1), (x2, y2), (180, 90, 255), 1)
                text_like += 1

        # 3) Motion mask (frame differencing)
        motion_count = 0
        if (not args.clean_ui) and prev_gray is not None and prev_gray.shape == gray.shape:
            diff = cv2.absdiff(gray, prev_gray)
            _, m = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
            m = cv2.medianBlur(m, 5)
            mc, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in mc:
                area = cv2.contourArea(c)
                if area < 90:
                    continue
                x, y, bw, bh = cv2.boundingRect(c)
                cv2.rectangle(view, (x, y), (x + bw, y + bh), (80, 255, 80), 1)
                cv2.putText(view, "motion", (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 255, 80), 1)
                motion_count += 1
                if motion_count >= 80:
                    break
        prev_gray = gray

        # 4) Visual keypoints (ORB) for feature richness
        kps: list[cv2.KeyPoint] = []
        if not args.clean_ui:
            orb = cv2.ORB_create(nfeatures=300)
            kps = orb.detect(gray, None)
            view = cv2.drawKeypoints(
                view,
                kps[:200],
                None,
                color=(255, 120, 0),
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
            )

        # 5) OCR word boxes (optional, interval-cached for smoother live refresh)
        ocr_count = 0
        now_t = time.perf_counter()
        do_ocr = (now_t - last_ocr_t) >= max(0.05, args.ocr_interval_s)
        if ocr_enabled and pytesseract is not None and do_ocr:
            try:
                roi_specs = [
                    (0, 0, w, int(h * 0.24)),
                    (0, int(h * 0.72), w, h - int(h * 0.72)),
                ]
                ocr_boxes: list[list[int]] = []
                ocr_scores: list[float] = []
                ocr_labels: list[str] = []
                for rx, ry, rw, rh in roi_specs:
                    roi = gray[ry : ry + rh, rx : rx + rw]
                    roi = cv2.resize(roi, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_CUBIC)
                    roi = cv2.GaussianBlur(roi, (3, 3), 0)
                    data = pytesseract.image_to_data(
                        roi,
                        output_type=pytesseract.Output.DICT,
                        config="--oem 3 --psm 6",
                    )
                    n = len(data.get("text", []))
                    for i in range(n):
                        text = (data["text"][i] or "").strip()
                        if not valid_ocr_text(text):
                            continue
                        conf_raw = data["conf"][i]
                        try:
                            conf = int(float(conf_raw))
                        except Exception:
                            conf = -1
                        if conf < args.ocr_min_conf:
                            continue

                        x = int(data["left"][i] / 1.8) + rx
                        y = int(data["top"][i] / 1.8) + ry
                        bw = int(data["width"][i] / 1.8)
                        bh = int(data["height"][i] / 1.8)
                        if bw < 10 or bh < 10:
                            continue
                        ocr_boxes.append([x, y, x + bw, y + bh])
                        ocr_scores.append(float(conf))
                        ocr_labels.append(f"{text} ({conf})")
                keep_ocr = nms_xyxy(ocr_boxes, ocr_scores, 0.25)
                max_ocr_boxes = min(args.max_ocr_boxes, 14) if args.clean_ui else args.max_ocr_boxes
                last_ocr_boxes = []
                last_ocr_labels = []
                for i in keep_ocr[:max_ocr_boxes]:
                    x1, y1, x2, y2 = ocr_boxes[i]
                    last_ocr_boxes.append([x1, y1, x2, y2])
                    last_ocr_labels.append(ocr_labels[i])
                last_ocr_t = now_t
            except Exception:
                # Keep analyzer running even if OCR backend fails on a frame.
                last_ocr_boxes = []
                last_ocr_labels = []
                last_ocr_t = now_t
        if ocr_enabled and last_ocr_boxes:
            for i, (x1, y1, x2, y2) in enumerate(last_ocr_boxes):
                cv2.rectangle(view, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(
                    view,
                    last_ocr_labels[i],
                    (x1, max(14, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
            ocr_count = len(last_ocr_boxes)

        # 6) Optional local Clash Royale detector model overlay (interval-cached)
        model_count = 0
        do_model = (now_t - last_model_t) >= max(0.1, args.model_interval_s)
        if yolo is not None and do_model:
            try:
                results = yolo.predict(source=frame, verbose=False, conf=0.35, iou=0.5)
                last_model_boxes = []
                last_model_labels = []
                if results:
                    r = results[0]
                    if hasattr(r, "boxes") and r.boxes is not None:
                        for b in r.boxes[:60]:
                            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                            conf = float(b.conf[0]) if b.conf is not None else 0.0
                            cls = int(b.cls[0]) if b.cls is not None else -1
                            name = r.names.get(cls, str(cls)) if hasattr(r, "names") else str(cls)
                            last_model_boxes.append((x1, y1, x2, y2))
                            last_model_labels.append(f"{name} {conf:.2f}")
                last_model_t = now_t
            except Exception:
                last_model_boxes = []
                last_model_labels = []
                last_model_t = now_t
        if last_model_boxes:
            for i, (x1, y1, x2, y2) in enumerate(last_model_boxes):
                cv2.rectangle(view, (x1, y1), (x2, y2), (0, 255, 120), 2)
                cv2.putText(
                    view,
                    last_model_labels[i],
                    (x1, max(14, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 120),
                    1,
                    cv2.LINE_AA,
                )
            model_count = len(last_model_boxes)

        # 7) Arena guide grid
        if not args.clean_ui:
            gx, gy = 8, 12
            for i in range(1, gx):
                x = int(i * w / gx)
                cv2.line(view, (x, 0), (x, h), (60, 60, 60), 1)
            for j in range(1, gy):
                y = int(j * h / gy)
                cv2.line(view, (0, y), (w, y), (60, 60, 60), 1)

        now = time.perf_counter()
        fps = 1.0 / max(1e-6, now - prev_t)
        prev_t = now
        draw_hud(
            view,
            [
                "OpenCV-only analyzer (no LLM actions)",
                f"resolution={w}x{h} fps={fps:.1f}",
                f"regions={obj_count} text_like={text_like} motion={motion_count} keypoints={min(len(kps), 200)} ocr={ocr_count} model={model_count}",
                f"ocr_enabled={int(ocr_enabled)} min_conf={args.ocr_min_conf} show_mser={int(args.show_mser)}",
                f"use_cr_model={int(yolo is not None)} path={args.cr_model_path}",
                f"clean_ui={int(args.clean_ui)}",
                f"capture={used_backend} monitor={args.monitor_index}",
                "q = quit",
            ],
        )

        cv2.imshow(args.window, view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        dt = time.perf_counter() - t0
        if dt < min_dt:
            time.sleep(min_dt - dt)

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
