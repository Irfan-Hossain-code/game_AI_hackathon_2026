#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-/tmp/clash_screen.png}"
adb exec-out screencap -p > "$OUT"
echo "$OUT"
