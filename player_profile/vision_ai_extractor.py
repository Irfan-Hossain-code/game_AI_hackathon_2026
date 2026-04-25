import json
import ollama


PROMPT = """
You are analyzing a Clash Royale gameplay screenshot.

Return only valid JSON.

Identify whether the player has just played a card or whether a clear card placement/action is visible.

Use this exact structure:

{
  "event_detected": true,
  "event": {
    "card": "card name or unknown",
    "lane": "left/right/middle/unknown",
    "side": "own_half/enemy_half/bridge/unknown",
    "action_type": "card_play/defense/push/spell/unknown",
    "confidence": 0.0,
    "evidence": "short visual reason"
  }
}

If no clear event is visible, return:

{
  "event_detected": false,
  "event": null
}

Rules:
- Return JSON only.
- Do not invent cards.
- Use unknown if unclear.
"""


def analyze_frame(image_path: str, timestamp: float) -> dict | None:
    response = ollama.chat(
        model="qwen2.5vl:3b",
        messages=[
            {
                "role": "user",
                "content": PROMPT,
                "images": [image_path],
            }
        ],
    )

    raw_text = response["message"]["content"]

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        print("Invalid JSON from Ollama:")
        print(raw_text)
        return None

    if not data.get("event_detected"):
        return None

    event = data["event"]

    return {
        "timestamp": timestamp,
        "card": event.get("card", "unknown"),
        "location": {
            "lane": event.get("lane", "unknown"),
            "side": event.get("side", "unknown"),
        },
        "context": event.get("action_type", "unknown"),
        "confidence": event.get("confidence", 0.0),
        "source": "ollama_qwen2.5vl",
        "evidence": event.get("evidence", ""),
    }