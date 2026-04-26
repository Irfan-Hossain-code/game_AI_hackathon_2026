#!/usr/bin/env python3
"""Minimal Kokoro smoke test (no Ollama). Run: python3 try_kokoro.py \"Your words here\""""

import sys

from kokoro_speech import speak_kokoro

if __name__ == "__main__":
    text = " ".join(sys.argv[1:]).strip() or "Hello from Kokoro on your machine."
    speak_kokoro(text)
