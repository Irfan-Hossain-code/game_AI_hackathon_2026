#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass


@dataclass
class Point:
    x: int
    y: int


def run_adb(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(["adb", *args], capture_output=True, text=True)


def adb_check() -> bool:
    r = run_adb(["get-state"])
    if r.returncode != 0:
        return False
    return "device" in (r.stdout or "")


def tap(x: int, y: int, dry_run: bool) -> bool:
    cmd = ["adb", "shell", "input", "tap", str(x), str(y)]
    if dry_run:
        print(f"[DRY] {' '.join(cmd)}")
        return True
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[ERR] tap failed: {r.stderr.strip()}")
        return False
    return True


def ask_point(name: str, width: int, height: int) -> Point:
    while True:
        raw = input(f"{name} (x y): ").strip()
        if not raw:
            continue
        parts = raw.replace(",", " ").split()
        if len(parts) != 2:
            print("Please enter two integers, e.g. 640 360")
            continue
        try:
            x = int(parts[0])
            y = int(parts[1])
        except ValueError:
            print("Invalid integers. Try again.")
            continue
        if not (0 <= x <= width - 1 and 0 <= y <= height - 1):
            print(f"Coordinates must be inside {width}x{height}.")
            continue
        return Point(x=x, y=y)


def confirm_tap(label: str, p: Point, dry_run: bool) -> None:
    ok = tap(p.x, p.y, dry_run=dry_run)
    if ok:
        print(f"[OK] tapped {label} at ({p.x},{p.y})")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Calibrate MuMu Clash Royale coordinates (1280x720 canvas)."
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print ADB commands without executing taps.",
    )
    ap.add_argument(
        "--out",
        default="mumu_coords.json",
        help="Output JSON file (default: mumu_coords.json)",
    )
    ap.add_argument("--width", type=int, default=1080, help="Canvas width (default: 1080)")
    ap.add_argument("--height", type=int, default=1920, help="Canvas height (default: 1920)")
    args = ap.parse_args()

    if args.width <= 0 or args.height <= 0:
        print("Width and height must be positive.")
        return 1

    if not args.dry_run and not adb_check():
        print("ADB device not ready. Make sure MuMu ADB is connected.")
        print("Try: adb devices")
        return 1

    print("\nMuMu calibration started.")
    print("Open Clash Royale battle screen first.")
    print(f"Enter coordinates in {args.width}x{args.height} space.\n")

    print("Step 1: card hand slots")
    slot0 = ask_point("slot0 center", args.width, args.height)
    confirm_tap("slot0", slot0, args.dry_run)
    slot1 = ask_point("slot1 center", args.width, args.height)
    confirm_tap("slot1", slot1, args.dry_run)
    slot2 = ask_point("slot2 center", args.width, args.height)
    confirm_tap("slot2", slot2, args.dry_run)
    slot3 = ask_point("slot3 center", args.width, args.height)
    confirm_tap("slot3", slot3, args.dry_run)

    print("\nStep 2: useful placement anchors")
    left_bridge = ask_point("left bridge", args.width, args.height)
    confirm_tap("left_bridge", left_bridge, args.dry_run)
    right_bridge = ask_point("right bridge", args.width, args.height)
    confirm_tap("right_bridge", right_bridge, args.dry_run)
    left_back_defense = ask_point("left back defense", args.width, args.height)
    confirm_tap("left_back_defense", left_back_defense, args.dry_run)
    right_back_defense = ask_point("right back defense", args.width, args.height)
    confirm_tap("right_back_defense", right_back_defense, args.dry_run)
    center_defense = ask_point("center defense", args.width, args.height)
    confirm_tap("center_defense", center_defense, args.dry_run)

    print("\nStep 3: menu/navigation anchors")
    print("Use Clash Royale home/result screens for these if possible.")
    battle_button = ask_point("battle button", args.width, args.height)
    confirm_tap("battle_button", battle_button, args.dry_run)
    ok_button = ask_point("ok/confirm button", args.width, args.height)
    confirm_tap("ok_button", ok_button, args.dry_run)
    play_again_button = ask_point("play again button", args.width, args.height)
    confirm_tap("play_again_button", play_again_button, args.dry_run)

    data = {
        "canvas": {"width": args.width, "height": args.height},
        "hand_slots": {
            "0": [slot0.x, slot0.y],
            "1": [slot1.x, slot1.y],
            "2": [slot2.x, slot2.y],
            "3": [slot3.x, slot3.y],
        },
        "anchors": {
            "left_bridge": [left_bridge.x, left_bridge.y],
            "right_bridge": [right_bridge.x, right_bridge.y],
            "left_back_defense": [left_back_defense.x, left_back_defense.y],
            "right_back_defense": [right_back_defense.x, right_back_defense.y],
            "center_defense": [center_defense.x, center_defense.y],
            "battle_button": [battle_button.x, battle_button.y],
            "ok_button": [ok_button.x, ok_button.y],
            "play_again_button": [play_again_button.x, play_again_button.y],
        },
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved calibration -> {args.out}")

    print("\nPaste this hand slot mapping into vision.py:")
    print("hand_slots = {")
    print(f"    0: ({slot0.x}, {slot0.y}),")
    print(f"    1: ({slot1.x}, {slot1.y}),")
    print(f"    2: ({slot2.x}, {slot2.y}),")
    print(f"    3: ({slot3.x}, {slot3.y}),")
    print("}")

    print("\nAnchor summary:")
    for k, v in data["anchors"].items():
        print(f"- {k}: ({v[0]}, {v[1]})")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
