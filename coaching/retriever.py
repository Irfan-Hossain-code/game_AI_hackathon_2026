"""
coaching/retriever.py

Searches Qdrant (vectors) and SQLite (meta decks) using a generated profile.
Combines textual strategy context and statistical context for the RAG Coach.
"""

import json
import os
import sqlite3

from mistralai.client import Mistral
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "knowledge.db")
QDRANT_PATH = os.path.join(os.path.dirname(__file__), "..", "qdrant_storage")

EMBED_MODEL = "mistral-embed"


def _get_mistral() -> Mistral:
    return Mistral(api_key=os.getenv("MISTRAL_API_KEY", ""))


def _get_qdrant() -> QdrantClient:
    return QdrantClient(path=QDRANT_PATH)


def search_knowledge(query: str, top_k: int = 3) -> dict:
    """Embeds the query and searches all vector collections."""
    mistral = _get_mistral()
    qdrant = _get_qdrant()
    
    # 1. Embed query
    response = mistral.embeddings.create(model=EMBED_MODEL, inputs=[query])
    query_vector = response.data[0].embedding
    
    # 2. Search collections
    collections = ["pro_tactics", "community_meta", "patch_knowledge"]
    results = {}
    
    for col in collections:
        try:
            hits = qdrant.search(
                collection_name=col,
                query_vector=query_vector,
                limit=top_k
            )
            results[col] = [{"score": hit.score, "text": hit.payload.get("text", "")} for hit in hits]
        except Exception as e:
            # Collection might not exist yet
            pass
            
    return results


def get_meta_deck_stats(deck_list: list) -> dict:
    """Check if the player's deck is in the meta decks SQLite table."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM meta_decks")
    decks = cursor.fetchall()
    conn.close()
    
    # Compare by sorting cards
    target_set = set(deck_list)
    
    for d in decks:
        try:
            cards = json.loads(d["cards_json"])
            if set(cards) == target_set:
                return {
                    "usage_rate": d["usage_rate"],
                    "win_rate": d["win_rate"],
                    "found": True
                }
        except:
            pass
            
    return {"found": False}
