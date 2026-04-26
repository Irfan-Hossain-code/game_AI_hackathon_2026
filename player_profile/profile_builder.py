import json
import os
from google import genai
from google.genai import types


client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def compact_api_features(api_features):
    return {
        "player_name": api_features.get("player_name"),
        "trophies": api_features.get("trophies"),
        "arena": api_features.get("arena"),
        "overall_win_rate": api_features.get("overall_win_rate"),
        "recent_win_rate": api_features.get("recent_win_rate"),
        "three_crown_rate": api_features.get("three_crown_rate"),
        "current_deck": api_features.get("current_deck"),
        "current_deck_average_elixir": api_features.get("current_deck_average_elixir"),
        "average_elixir_leaked": api_features.get("average_elixir_leaked"),
    }


def compact_gameplay_features(gameplay_features):
    return {
        "opening_style": gameplay_features.get("opening_style"),
        "pressure_style": gameplay_features.get("pressure_style"),
        "event_count": gameplay_features.get("event_count"),
        "own_half_plays": gameplay_features.get("own_half_plays"),
        "enemy_half_plays": gameplay_features.get("enemy_half_plays"),
        "lane_counts": gameplay_features.get("lane_counts"),
        "evidence": gameplay_features.get("evidence", [])[:6],
        "raw_events": gameplay_features.get("raw_events", [])[:12],
    }


def build_ai_profile(api_features, gameplay_features):
    compact_api = compact_api_features(api_features)
    compact_gameplay = compact_gameplay_features(gameplay_features)

    prompt = f"""
You are a Clash Royale coaching profile builder.

Create a useful player profile for a coach AI.

Return ONLY valid JSON. No markdown.

Use exactly:

{{
  "playstyle_summary": "",
  "player_archetype": "",
  "strengths": [],
  "weaknesses": [],
  "habits": [],
  "latest_match_observations": [],
  "coaching_focus": [],
  "trust_notes": []
}}

Rules:
- Max 4 items per list.
- Be concise.
- Use evidence from the data.
- API stats describe general profile.
- Gameplay events describe latest match only.
- Mention uncertainty if video extraction looks noisy.

API_FEATURES:
{json.dumps(compact_api, indent=2)}

GAMEPLAY_FEATURES:
{json.dumps(compact_gameplay, indent=2)}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
        ),
    )

    return json.loads(response.text)