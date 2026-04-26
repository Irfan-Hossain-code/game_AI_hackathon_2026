import json
import vertexai
from vertexai.generative_models import GenerativeModel, Part, GenerationConfig


PROJECT_ID = "clash-royal-ai"
LOCATION = "global"

vertexai.init(project=PROJECT_ID, location=LOCATION)

model = GenerativeModel("gemini-2.5-flash")


PROMPT = """
You are analyzing one Clash Royale gameplay screenshot.

Your task is to detect whether the player has JUST played a card.

Return ONLY valid JSON.

Use exactly this format:

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

If no clear new card play is visible, return:

{
  "event_detected": false,
  "event": null
}

Rules:
- Be strict.
- Do not count old troops walking as a new card play.
- Do not invent cards.
- Prefer "unknown" if the card is unclear.
- Return JSON only.
"""


def analyze_frame(image_path: str, timestamp: float):
    with open(image_path, "rb") as f:
        image_part = Part.from_data(
            data=f.read(),
            mime_type="image/jpeg"
        )

    response = model.generate_content(
        [PROMPT, image_part],
        generation_config=GenerationConfig(
            temperature=0.0,
            response_mime_type="application/json"
        )
    )

    try:
        data = json.loads(response.text)
    except json.JSONDecodeError:
        print("Invalid JSON from Gemini:")
        print(response.text)
        return None

    if not data.get("event_detected"):
        return None

    event = data.get("event") or {}

    return {
        "timestamp": timestamp,
        "card": event.get("card", "unknown"),
        "location": {
            "lane": event.get("lane", "unknown"),
            "side": event.get("side", "unknown")
        },
        "context": event.get("action_type", "unknown"),
        "confidence": event.get("confidence", 0.0),
        "source": "vertex_gemini_adc",
        "evidence": event.get("evidence", "")
    }