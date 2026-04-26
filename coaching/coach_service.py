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


def build_coaching_prompt(profile: dict, knowledge: dict, deck_stats: dict) -> str:
    """Assemble the Mega-Prompt for Mistral."""
    
    # Extract player details
    name = profile.get("player_name", "Unknown Player")
    deck = profile.get("features", {}).get("current_deck", [])
    playstyle = profile.get("playstyle", "balanced")
    weaknesses = profile.get("weaknesses", [])
    strengths = profile.get("strengths", [])
    
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

TASK:
Based strictly on the data above, provide {name} with 3 specific, highly targeted bullet points on how to improve their gameplay. Do NOT hallucinate card interactions that are not fundamentally true to Clash Royale. Focus directly on their weaknesses and their specific deck composition.
"""
    return prompt


def run_coaching(player_tag: str) -> str:
    print(f"\n--- Initialising Coaching Session for {player_tag} ---")
    
    # 1. Fetch live features & playstyle
    print("[1] Building player profile from Clash Royale API...")
    profile = build_player_profile(player_tag)
    
    # 2. Build search query for vector DB
    print("[2] Extracting vector search context from profile...")
    weak = " ".join(profile.get("weaknesses", []))
    deck = " ".join(profile.get("features", {}).get("current_deck", []))
    
    # What do we need to search for? 
    # E.g. "How to stop leaking elixir, tips for using hog rider..."
    query = f"Tactics for {deck} deck. How to fix weaknesses: {weak}"
    
    # 3. Retrieve
    print(f"[3] Searching Qdrant knowledge base for query: '{query[:50]}...'")
    knowledge = search_knowledge(query, top_k=2)
    deck_stats = get_meta_deck_stats(profile.get("features", {}).get("current_deck", []))
    
    # 4. Generate Prompt
    prompt = build_coaching_prompt(profile, knowledge, deck_stats)
    
    # 5. Mistral Generation
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
    import sys
    if len(sys.argv) > 1:
        tag = sys.argv[1]
    else:
        tag = "#VR08VQLL2"
    
    run_coaching(tag)
