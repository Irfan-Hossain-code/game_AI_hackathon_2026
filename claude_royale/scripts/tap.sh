#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: tap.sh <x> <y>"
  exit 1
fi

adb shell input tap "$1" "$2"
