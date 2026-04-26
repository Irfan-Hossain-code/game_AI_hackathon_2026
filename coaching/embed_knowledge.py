"""
coaching/embed_knowledge.py

Synchronises the SQLite 'raw_chunks' table (populated by Node scrapers)
into the local Qdrant vector database using Mistral embeddings.
Only processes chunks where embedded=0, preventing redundant API calls.
"""

import json
import os
import sqlite3
import time
import uuid

from mistralai.client import Mistral
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "knowledge.db")
QDRANT_PATH = os.path.join(os.path.dirname(__file__), "..", "qdrant_storage")

EMBED_MODEL = "mistral-embed"
BATCH_SIZE = 64
VECTOR_DIM = 1024


def _get_mistral() -> Mistral:
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY not found in .env")
    return Mistral(api_key=api_key)


def _init_qdrant() -> QdrantClient:
    """Initialise local file-based Qdrant client."""
    client = QdrantClient(path=QDRANT_PATH)
    return client


def _ensure_collection(client: QdrantClient, name: str) -> None:
    """Create collection if it doesn't exist."""
    collections = [c.name for c in client.get_collections().collections]
    if name not in collections:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE)
        )
        print(f"[qdrant] Created collection '{name}'")


def get_embeddings(client: Mistral, texts: list[str]) -> list[list[float]]:
    """Fetch Mistral embeddings for a list of text strings."""
    if not texts:
        return []
    
    response = client.embeddings.create(model=EMBED_MODEL, inputs=texts)
    return [item.embedding for item in response.data]


def sync_chunks() -> None:
    """Read unembedded chunks from SQLite, embed, store in Qdrant, mark as embedded."""
    mistral = _get_mistral()
    qdrant = _init_qdrant()
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM raw_chunks WHERE embedded = 0")
    rows = cursor.fetchall()
    
    if not rows:
        print("[sync] No new raw_chunks to embed.")
        return
        
    print(f"[sync] Found {len(rows)} unembedded chunks.")
    
    # Process in batches
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i : i + BATCH_SIZE]
        texts = [row["text"] for row in batch]
        collections = [row["collection"] for row in batch]
        ids = [row["id"] for row in batch]
        
        # Parse payloads
        payloads = []
        for row in batch:
            try:
                p = json.loads(row["payload_json"])
            except:
                p = {}
            p["text"] = row["text"]
            p["scraped_at"] = row["scraped_at"]
            payloads.append(p)
            
        print(f"[sync] Embedding batch {i // BATCH_SIZE + 1} ({len(batch)} chunks)...")
        vectors = get_embeddings(mistral, texts)
        
        # Group by collection so we can upsert smoothly
        collection_points = {}
        for idx in range(len(batch)):
            col = collections[idx]
            vec = vectors[idx]
            pay = payloads[idx]
            
            if col not in collection_points:
                collection_points[col] = []
                
            collection_points[col].append(PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload=pay
            ))
            
        # Upsert
        for col, points in collection_points.items():
            _ensure_collection(qdrant, col)
            qdrant.upsert(collection_name=col, points=points)
            
        # Mark as embedded in SQLite
        id_list = ",".join("?" * len(ids))
        cursor.execute(f"UPDATE raw_chunks SET embedded = 1 WHERE id IN ({id_list})", ids)
        conn.commit()
        
        time.sleep(1) # respectful rate limiting
        
    print("[sync] All chunks embedded safely!")
    conn.close()


if __name__ == "__main__":
    sync_chunks()
