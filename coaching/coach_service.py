"""
coaching/coach_service.py

Connects player_profile creation with the RAG retrieving layer and executes
a prompt against Mistral to generate personalized coaching advice.
"""

import json
import os
import sys

from mistralai.client import Mistral
from dotenv import load_dotenv

# Add paths to allow imports from other modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "player_profile")))

from player_profile.profile_service import build_player_profile
from coaching.retriever import search_knowledge, get_meta_deck_stats

load_dotenv()

def load_events_from_json(path: str) -> list:
    with open(path, "r") as f:
        return json.load(f)

def load_ai_profile_from_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

# adjusting json calls
def events_to_profile(events: list) -> dict:
    cards = [e["card"] for e in events if e.get("card") and e["card"] != "unknown"]

    # simple frequency-based deck guess
    deck = list(set(cards))[:8]

    return {
        "player_name": "Video Player",
        "playstyle": "unknown",
        "features": {
            "current_deck": deck
        },
        "weaknesses": extract_basic_weaknesses(events),
        "strengths": []
    }

def extract_basic_weaknesses(events):
    if len(events) < 5:
        return ["Low activity / passive play"]

    spells = [e for e in events if e.get("context") == "spell"]
    if len(spells) > len(events) * 0.5:
        return ["Over-reliance on spells"]

    return []


def build_coaching_prompt(profile: dict, knowledge: dict, deck_stats: dict, ai_profile: dict) -> str:
    """Assemble the Mega-Prompt for Mistral."""
    
    # Extract player details
    name = profile.get("player_name", "Unknown Player")
    deck = profile.get("features", {}).get("current_deck", [])
    playstyle = profile.get("playstyle", "balanced")
    weaknesses = profile.get("weaknesses", [])
    strengths = profile.get("strengths", [])
    ai_summary = ai_profile.get("playstyle_summary", "")
    ai_archetype = ai_profile.get("player_archetype", "")
    ai_strengths = ai_profile.get("strengths", [])
    ai_weaknesses = ai_profile.get("weaknesses", [])
    ai_habits = ai_profile.get("habits", [])
    ai_observations = ai_profile.get("latest_match_observations", [])
    ai_focus = ai_profile.get("coaching_focus", [])
    ai_notes = ai_profile.get("trust_notes", [])
    
    # Extract structural RAG texts
    tactics = ""
    for item in knowledge.get("pro_tactics", []):
        tactics += f"- {item['text']}\n"
        
    community = ""
    for item in knowledge.get("community_meta", []):
        community += f"- {item['text']}\n"
        
    patch = ""
    for item in knowledge.get("patch_knowledge", []):
        patch += f"- {item['text']}\n"
        
    # Format the prompt
    prompt = f"""You are an elite Clash Royale AI Coach. You speak concisely and with high strategic authority.

PLAYER PROFILE:
- Name: {name}
- Playstyle: {playstyle}
- Current Deck: {', '.join(deck)}
- Strengths: {', '.join(strengths) if strengths else "None noted"}
- Weaknesses: {', '.join(weaknesses) if weaknesses else "None noted"}

"""
    if deck_stats.get("found"):
        prompt += f"DECK META STATS:\nThis exact deck has a {deck_stats['win_rate']}% win rate globally.\n\n"
    else:
        prompt += "DECK META STATS:\nThis is an off-meta or custom deck. No clean win rate available.\n\n"
        
        prompt += f"""STRATEGIC KNOWLEDGE BASE (Use this to ground your advice):
== Pro Tactics ==
{tactics if tactics else 'No pro tactics retrieved.'}

== Community Meta Sentiment ==
{community if community else 'No community data retrieved.'}

== Recent Patch Note Context ==
{patch if patch else 'No patch notes retrieved.'}
"""

    prompt += f"""
AI VIDEO PLAYER PROFILE:
Summary: {ai_summary}
Archetype: {ai_archetype}

Strengths:
{chr(10).join(f"- {s}" for s in ai_strengths) if ai_strengths else "- None"}

Weaknesses:
{chr(10).join(f"- {w}" for w in ai_weaknesses) if ai_weaknesses else "- None"}

Habits:
{chr(10).join(f"- {h}" for h in ai_habits) if ai_habits else "- None"}

Latest Match Observations:
{chr(10).join(f"- {o}" for o in ai_observations) if ai_observations else "- None"}

Coaching Focus:
{chr(10).join(f"- {c}" for c in ai_focus) if ai_focus else "- None"}

Trust Notes:
{chr(10).join(f"- {n}" for n in ai_notes) if ai_notes else "- None"}

TASK:
Return the coaching report using EXACTLY this structure:

## Summary
Briefly summarize the player.

## Archetype
State the player archetype and what it means.

## Habits
List the main gameplay habits.

## Advice
Give 3 specific actionable coaching points.

## Strengths
List the strongest parts of the player gameplay.

## Weaknesses
List the main weaknesses.

## Notes
Mention reliability/trust notes, especially if video observations are based on limited matches.

Do NOT hallucinate card interactions that are not fundamentally true to Clash Royale.
Use the exact deck, API profile, video events, and AI video player profile together.
"""
    return prompt


def run_coaching(
    player_tag: str = "#VR08VQLL2",
    events_path: str = "player_profile/video_events.json",
    ai_profile_path: str = "player_profile/ai_player_profile.json"
) -> str:
    print(f"\n--- Initialising Coaching Session for {player_tag} ---")

    # 1. Build base profile from Clash Royale API
    print("[1] Building player profile from Clash Royale API...")
    profile = build_player_profile(player_tag)

    # 2. Load video events
    print("[1b] Loading video events...")
    events = load_events_from_json(events_path)

    # 3. Add video-derived information to existing profile
    print("[1c] Loading AI player profile...")
    ai_profile = load_ai_profile_from_json(ai_profile_path)

    video_weaknesses = extract_basic_weaknesses(events)

    existing_weaknesses = profile.get("weaknesses", [])
    existing_strengths = profile.get("strengths", [])

    ai_strengths = ai_profile.get("strengths", [])
    ai_weaknesses = ai_profile.get("weaknesses", [])

    profile["weaknesses"] = list(set(existing_weaknesses + video_weaknesses + ai_weaknesses))
    profile["strengths"] = list(set(existing_strengths + ai_strengths))
    profile["playstyle"] = ai_profile.get("player_archetype", profile.get("playstyle", "balanced"))
    profile["playstyle_summary"] = ai_profile.get("playstyle_summary", "")
    profile["habits"] = ai_profile.get("habits", [])
    profile["coaching_focus"] = ai_profile.get("coaching_focus", [])
    profile["trust_notes"] = ai_profile.get("trust_notes", [])
    profile["video_events"] = events

    # 4. Build search query for vector DB
    print("[2] Extracting vector search context from profile...")

    weak = " ".join(profile.get("weaknesses", []))
    strengths = " ".join(profile.get("strengths", []))
    playstyle = profile.get("playstyle", "balanced")
    deck = " ".join(profile.get("features", {}).get("current_deck", []))

    query = (
        f"Tactics for {deck} deck. "
        f"Player playstyle: {playstyle}. "
        f"Strengths: {strengths}. "
        f"How to fix weaknesses: {weak}"
    )

    # 5. Retrieve RAG knowledge
    print(f"[3] Searching Qdrant knowledge base for query: '{query[:80]}...'")
    knowledge = search_knowledge(query, top_k=2)

    try:
        deck_stats = get_meta_deck_stats(
            profile.get("features", {}).get("current_deck", [])
        )
    except Exception as e:
        print(f"[WARN] Could not load deck stats: {e}")
        deck_stats = {"found": False}

    # 6. Generate coaching prompt
    prompt = build_coaching_prompt(profile, knowledge, deck_stats, ai_profile)

    # 7. Mistral generation
    print("[4] Generating personalised advice via Mistral LLM...")

    mistral = Mistral(api_key=os.getenv("MISTRAL_API_KEY", ""))

    response = mistral.chat.complete(
        model="mistral-large-latest",
        messages=[{"role": "user", "content": prompt}]
    )

    advice = response.choices[0].message.content

    print("\n--- COACHING ADVICE ---\n")
    print(advice)
    print("\n-----------------------\n")

    return advice


if __name__ == "__main__":
    if len(sys.argv) > 1:
        tag = sys.argv[1]
    else:
        tag = "#VR08VQLL2"

    if len(sys.argv) > 2:
        events_path = sys.argv[2]
    else:
        events_path = "player_profile/video_events.json"

    if len(sys.argv) > 3:
        ai_profile_path = sys.argv[3]
    else:
        ai_profile_path = "player_profile/ai_player_profile.json"

    run_coaching(tag, events_path, ai_profile_path)