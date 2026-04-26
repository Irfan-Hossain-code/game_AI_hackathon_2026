#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: play_card.sh <slot> <x> <y>"
  exit 1
fi

SLOT="$1"
X="$2"
Y="$3"

python3 - "$SLOT" "$X" "$Y" <<'PY'
import json
import os
import sys
from pathlib import Path

slot = int(sys.argv[1])
x = int(sys.argv[2])
y = int(sys.argv[3])
coords = Path("claude_royale/config/coordinates.json")
data = json.loads(coords.read_text(encoding="utf-8"))
hand = data.get("hand_slots", {})
xy = hand.get(str(slot))
if not xy:
    raise SystemExit(f"missing slot {slot}")
sx, sy = int(xy[0]), int(xy[1])
os.system(f"adb shell input tap {sx} {sy}")
os.system(f"adb shell input tap {x} {y}")
print(f"played slot={slot} to {x},{y}")
PY
