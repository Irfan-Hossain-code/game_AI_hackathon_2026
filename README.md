# Clash Royale AI Coach (MuMu + Ollama + Kokoro)

This project is a local, voice-enabled Clash Royale AI coach/teammate for hackathon use.

It runs on macOS with MuMu Player, uses:
- a fast Ollama model for decisions/coaching text, and
- an optional vision-capable model for screenshot navigation fallback,
reuses Kokoro for TTS, and supports voice commands via STT.

## Project Goal

Build a low-latency agent that can:

- Watch the live game screen from MuMu.
- Track opponent state (cards seen, rough elixir estimate, key counters like Log).
- Decide if and where to place cards.
- Speak short tactical callouts in real time (what + where + why).
- Accept voice questions/commands from the player.

The architecture style is inspired by Claude-Royale's loop-driven game agent concept, adapted to this local Python stack:

- [claude-royale reference](https://github.com/houseworthe/claude-royale)

## Current Architecture (Implemented)

- `main.py`
  - Main orchestrator loop.
  - Commander loop keeps navigating between matches and battling continuously.
  - Runs state polling, decision, action, and speech dispatch.
  - Starts STT listener thread and non-blocking speech queue.
- `vision.py`
  - MuMu screen capture (`mss`).
  - ADB control (`adb shell input tap ...`).
  - Coordinate handling for `1080x1920`.
  - Loads calibrated hand slots from `mumu_coords.json` if present.
  - Loads navigation anchors (`battle_button`, `ok_button`, `play_again_button`).
  - Tracks lightweight enemy memory signals.
- `navigator.py`
  - Minimal commander-style navigation.
  - Launches Clash Royale package.
  - Classifies screen state (`menu`/`popup`/`in_battle`/`unknown`) and chooses next action.
  - Optional screenshot-to-LLM nav fallback (`USE_LLM_NAV=1`).
  - Runs continuously, not just once at startup.
- `screen_classifier.py`
  - Fast deterministic screen classifier for nav state machine.
- `brain.py`
  - Ollama HTTP client (`/api/generate`) using one fast model.
  - Produces action plan JSON.
  - Produces short spoken narration.
  - Graceful fallback on model timeouts/errors.
- `kokoro_speech.py`
  - Existing Kokoro TTS path (reused).
  - No F5 dependency in active runtime.
- `stt_utils.py`
  - Push-to-talk transcription utility extracted for cleaner runtime path.

## Runtime Flow

1. Poll frame from MuMu.
2. Extract lightweight game signals and update enemy memory.
3. Send compact state to Ollama model for action plan.
4. If plan says play, send ADB taps (card slot then target).
5. Generate spoken coaching line and send to voice queue.
6. STT thread captures player voice commands and feeds next decision ticks.

## Key Features

- Fast model strategy (`FAST_MODEL`) for battle decisions and narration.
- Optional dedicated vision model (`VISION_MODEL`) for screenshot navigation fallback.
- Non-blocking voice queue with stale-line dropping (keeps gameplay responsive).
- ADB safety:
  - Dry-run mode (`DRY_RUN=1`).
  - Tap rate-limiting.
- Latency logs for capture/extract/LLM.
- 1080x1920 MuMu support.

## Requirements

- macOS
- MuMu Player (Android device set to 1080x1920 in current setup)
- Python 3.10+
- Ollama running locally
- ADB in shell PATH
- `espeak-ng` for Kokoro English phonemization

## Install

```bash
cd /Users/mohammadasender/Desktop/game_AI_hackathon_2026
pip3 install -r requirements.txt
brew install espeak-ng
```

## Setup

1. Start Ollama:

```bash
ollama serve
```

2. Verify model exists:

```bash
ollama list
```

3. Verify ADB works:

```bash
adb devices
```

4. Calibrate tap coordinates (battle + menu):

```bash
python3 calibrate_mumu.py --width 1080 --height 1920
```

This writes `mumu_coords.json`.

## Run

Dry run (safe, no real taps):

```bash
DRY_RUN=1 FAST_MODEL=gemma2:2b OLLAMA_TIMEOUT_S=8 python3 main.py
```

Live run (real taps):

```bash
FAST_MODEL=gemma2:2b OLLAMA_TIMEOUT_S=8 python3 main.py
```

## Environment Variables

- `FAST_MODEL` (default: `gemma2:2b`)
- `OLLAMA_TIMEOUT_S` (default: `8`)
- `DRY_RUN` (`1` for no real tap execution)
- `USE_ADB_SCREENSHOT` (default: `1`; capture from emulator framebuffer, not desktop window)
- `AUTO_NAVIGATE` (`1` default, `0` to disable auto-launch/navigation)
- `CR_PACKAGE` (default: `com.supercell.clashroyale`)
- `USE_LLM_NAV` (`1` default; set `0` to force deterministic-only nav)
- `VISION_MODEL` (default: `gemma4:e4b`, preferred for screenshot nav tasks)
- `NAV_TIMEOUT_S` (default: `8`, timeout for screenshot nav model calls)
- `SHOW_TOUCHES` (default: `1`; show touch dots inside MuMu during taps)
- `POINTER_LOCATION` (default: `0`; show pointer path/coordinates inside MuMu)
- `MODEL_OVERLAY` (default: `1`; show live agent HUD window with cursor + detected elements)
- `MODEL_DETECTIONS` (default: `1`; ask Gemma4 for live detection boxes used by overlay)
- `PERCEPTION_INTERVAL_S` (default: `1.2`; how often model detections refresh)
- `NAV_MODEL` (legacy alias; used if `VISION_MODEL` is not set)

## Important Notes

- If `adb` is missing, install Android platform-tools and add it to PATH.
- If taps are inaccurate, rerun calibration.
- If MuMu resolution/layout changes, recalibrate.
- STT push-to-talk uses `space` in the current runtime path.

## Repository Files (Core)

- `main.py`
- `vision.py`
- `brain.py`
- `kokoro_speech.py`
- `stt_utils.py`
- `calibrate_mumu.py`
- `requirements.txt`

## Hackathon Scope Summary

This is a practical MVP focused on:

- local execution,
- low-latency response,
- human-agent voice interaction,
- game control safety,
- and resilient restartability.

The project is intentionally modular so future work (better vision detector, richer deck logic, stronger memory model, improved narration policy) can be added without rewriting the full loop.

## Legacy backend modules

This repo also contains prior backend/RAG ingestion components (SQLite/Qdrant/Mistral pipeline and Node ingestion scripts) from earlier work. Keep using those paths if you need the old profile/coaching data flow.
