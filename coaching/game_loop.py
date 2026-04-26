"""
coaching/game_loop.py

The main event loop. Connects the CV Radar output securely to 
the LLM Brain using RAG, producing 1-sentence reactive coaching.
"""

import os
import sys

from mistralai.client import Mistral
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "player_profile")))

from player_profile.cv_radar_engine import CVRadarEngine
from coaching.retriever import search_knowledge

load_dotenv()

def react_to_game_state(state):
    """
    RAG integration and single-sentence generation logic.
    Receives JSON from OpenCV.
    """
    ts = state["timestamp"]
    action = state["opponent_action"]
    loc = state["location"]
    hand = state["user_hand"]
    elixir = state["user_elixir"]
    
    # The pure English bridge
    constructed_event = f"At {ts}, the opponent {action} on the {loc}."
    
    # 1. RAG Retrieve Tactics for the Opponent's card
    # Example action string: "Placed Goblin Barrel" -> split and search "Goblin Barrel"
    card_name = action.replace("Placed ", "")
    query = f"Defense against {card_name} using cards {', '.join(hand)}"
    knowledge = search_knowledge(query, top_k=1)
    
    # Flatten tactics
    tactics = ""
    for item in knowledge.get("pro_tactics", []):
        tactics += f"- {item['text']}\n"
    for item in knowledge.get("community_meta", []):
        tactics += f"- {item['text']}\n"

    # 2. Build the exact prompt structure the user described
    prompt = f"""You are an expert Clash Royale coach. I will give you a real-time game state. Analyze the opponent's action, look at the user's current hand and elixir, and output a short, 1-sentence instruction on what the user should do next.

PRO TACTICS (Use to guide your advice):
{tactics if tactics else "No specific tactics found."}

GAME STATE:
{constructed_event} The user has {elixir} elixir and these cards in hand: {', '.join(hand)}."""

    # 3. Mistral Call
    mistral = Mistral(api_key=os.getenv("MISTRAL_API_KEY", ""))
    response = mistral.chat.complete(
        model="mistral-large-latest",
        messages=[{"role": "user", "content": prompt}]
    )
    
    print("\n---------------------------------------------------------")
    print(f"[{ts}] {action} ({loc}) | Elixir: {elixir} | Hand: {hand[0]}, {hand[1]}...")
    print(f">> COACH: {response.choices[0].message.content.strip()}")
    print("---------------------------------------------------------\n")


def main():
    print("Starting OpenCV Radar & AI Coaching Loop...")
    radar = CVRadarEngine()
    
    # As the radar yields events (60fps simulation), we react instantly
    for state_json in radar.run_radar_loop():
        react_to_game_state(state_json)


if __name__ == "__main__":
    main()
