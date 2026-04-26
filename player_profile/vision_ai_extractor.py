import json
import os
from google import genai
from google.genai import types


client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


PROMPT = """
You are analyzing multiple Clash Royale gameplay screenshots sampled every few seconds.

Each image has a timestamp before it.

Important:
- The PLAYER is the bottom-side player.
- The OPPONENT is the top-side player.
- Own half means bottom half.
- Enemy half means top half.

Return ONLY valid JSON.

Return this exact structure:

{
  "frames": [
    {
      "timestamp": 0.0,
      "player_actions": [
        {
          "actor": "player",
          "action": "played Knight on left lane in own half",
          "card": "Knight",
          "lane": "left/right/middle/unknown",
          "side": "own_half/enemy_half/bridge/unknown",
          "action_type": "card_play/defense/push/spell/unknown",
          "elixir_amount": 0,
          "confidence": 0.0,
          "evidence": "short visual reason"
        }
      ],
      "opponent_actions": [
        {
          "actor": "opponent",
          "action": "played Giant on right lane in enemy half",
          "card": "Giant",
          "lane": "left/right/middle/unknown",
          "side": "own_half/enemy_half/bridge/unknown",
          "action_type": "card_play/defense/push/spell/unknown",
          "confidence": 0.0,
          "evidence": "short visual reason"
        }
      ]
    }
  ]
}

Rules:
- Separate player actions and opponent actions.
- Only include actions that are newly visible or strongly suggested in that frame.
- Do not list old troops just walking.
- If unsure who played it, do not include it.
- Prefer "unknown" over guessing.
- Return JSON only.
"""


def analyze_frames_batch(frame_items):
    contents = [PROMPT]

    for item in frame_items:
        timestamp = item["timestamp"]
        image_path = item["path"]

        contents.append(f"Timestamp: {timestamp}s")

        with open(image_path, "rb") as f:
            contents.append(
                types.Part.from_bytes(
                    data=f.read(),
                    mime_type="image/jpeg",
                )
            )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
        ),
    )

    try:
        data = json.loads(response.text)
    except json.JSONDecodeError:
        print("Invalid JSON from Gemini:")
        print(response.text)
        return []

    
    events = []

    for frame in data.get("frames", []):
        timestamp = frame.get("timestamp")

        for action in frame.get("player_actions", []):
            events.append({
                "timestamp": timestamp,
                "actor": "player",
                "action": action.get("action", ""),
                "card": action.get("card", "unknown"),
                "location": {
                    "lane": action.get("lane", "unknown"),
                    "side": action.get("side", "unknown"),
                },
                "context": action.get("action_type", "unknown"),
                "elixir_amount": action.get("elixir_amount"),
                "confidence": action.get("confidence", 0.0),
                "source": "gemini_api_key_batch",
                "evidence": action.get("evidence", ""),
            })

        for action in frame.get("opponent_actions", []):
            events.append({
                "timestamp": timestamp,
                "actor": "opponent",
                "action": action.get("action", ""),
                "card": action.get("card", "unknown"),
                "location": {
                    "lane": action.get("lane", "unknown"),
                    "side": action.get("side", "unknown"),
                },
                "context": action.get("action_type", "unknown"),
                "elixir_amount": action.get("elixir_amount"),
                "confidence": action.get("confidence", 0.0),
                "source": "gemini_api_key_batch",
                "evidence": action.get("evidence", ""),
            })

    return events