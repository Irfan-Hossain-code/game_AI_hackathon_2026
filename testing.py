"""Quick F5-TTS (MLX) test: edit PROMPT (and optional ref) then run `python3 testing.py`."""

from pathlib import Path

from f5_mlx_compat import speak_f5

# ----- edit this -----
PROMPT = "Giant on the left bridge. Let's go!"
# Your reference clip (must be 24 kHz mono WAV). If missing, the script uses the
# bundled English sample and its fixed transcript.
REF_AUDIO = "ref.wav"
REF_TEXT = "This is the reference voice speaking."
STEPS = 4
MODEL = "alandao/f5-tts-mlx-4bit"
# ---------------------


def main() -> None:
    if not Path(REF_AUDIO).is_file() and REF_AUDIO:
        print(f"No file at {REF_AUDIO!r} — using built-in reference audio.\n")

    speak_f5(
        PROMPT,
        ref_audio=REF_AUDIO,
        ref_text=REF_TEXT,
        hf_model=MODEL,
        steps=STEPS,
    )


if __name__ == "__main__":
    main()
