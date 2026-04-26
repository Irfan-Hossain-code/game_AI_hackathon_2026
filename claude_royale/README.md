# Claude Royale Mode (MuMu + Ollama)

This is a direct `claude-royale` style harness for this repo:

- Commander loop handles menus and battle lifecycle.
- 3 parallel player workers run during battle.
- Each worker reads live visuals and proposes card actions.
- Worker actions are played through ADB taps.

## Run

```bash
cd /Users/mohammadasender/Desktop/game_AI_hackathon_2026
python3 claude_royale/commander.py
```

Voice callouts:

```bash
python3 claude_royale/commander.py --voice
```

Dry run (no real taps):

```bash
DRY_RUN=1 python3 claude_royale/commander.py
```

## Files

- `config/coordinates.json` - calibrated hand slots + UI anchors
- `config/gameplay.json` - target placement points
- `memory/*` - persistent notes/status/goals
- `scripts/screenshot.sh` - ADB screenshot helper
- `scripts/tap.sh` - ADB tap helper
- `scripts/play_card.sh` - slot->target play helper
- `scripts/watch-agent.sh` - launcher wrapper
