"""
storage/vector_db.py

Qdrant vector database wrapper.
Manages 4 collections for the knowledge pipeline:
  - pro_tactics      : YouTube pro player transcripts
  - community_meta   : Reddit / Discord / X community signals
  - patch_knowledge  : Supercell blog balance changes
  - discord_trends   : Real-time Discord competitive channel data

Runs in-memory by default (no Qdrant server needed for hackathon).
Set QDRANT_HOST in .env to point at a running server instead.
"""

import os
import uuid
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)

load_dotenv()

# ── Client Setup ─────────────────────────────────────────────────────────────

QDRANT_HOST = os.getenv("QDRANT_HOST", "").strip()
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
VECTOR_DIM = 1024  # Mistral embed-large dimension

def _build_client() -> QdrantClient:
    if QDRANT_HOST:
        print(f"[qdrant] Connecting to server at {QDRANT_HOST}:{QDRANT_PORT}")
        return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    print("[qdrant] Running in-memory mode (no QDRANT_HOST set)")
    return QdrantClient(":memory:")

client = _build_client()


# ── Collection Definitions ────────────────────────────────────────────────────

COLLECTIONS = {
    "pro_tactics": {
        "description": "YouTube transcripts from CRL and pro player channels",
        "payload_fields": ["video_id", "channel", "start_time", "text", "scraped_at", "tags"],
    },
    "community_meta": {
        "description": "Reddit posts, Discord messages, X/Twitter — community strategy signals",
        "payload_fields": ["platform", "post_id", "score", "text", "scraped_at", "tags"],
    },
    "patch_knowledge": {
        "description": "Supercell blog balance changes and patch notes",
        "payload_fields": ["patch_version", "publish_date", "cards_affected", "text", "scraped_at"],
    },
    "discord_trends": {
        "description": "Real-time competitive Discord channel discussions",
        "payload_fields": ["server", "channel", "author", "text", "scraped_at"],
    },
}


# ── Init ──────────────────────────────────────────────────────────────────────

def init_collections() -> None:
    """Create all Qdrant collections if they don't already exist."""
    existing = {c.name for c in client.get_collections().collections}

    for name in COLLECTIONS:
        if name in existing:
            print(f"[qdrant] Collection already exists: {name}")
            continue
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )
        print(f"[qdrant] Created collection: {name}")


# ── Upsert ────────────────────────────────────────────────────────────────────

def upsert_points(
    collection: str,
    texts: list[str],
    vectors: list[list[float]],
    payloads: list[dict[str, Any]],
) -> int:
    """
    Insert or update points in a Qdrant collection.

    Args:
        collection:  Name of the target collection.
        texts:       Raw text for each point (stored in payload as 'text').
        vectors:     Pre-computed embedding vectors.
        payloads:    Metadata dicts — one per point.

    Returns:
        Number of points upserted.
    """
    if not texts:
        return 0

    now = datetime.now(timezone.utc).isoformat()
    points = []

    for text, vector, payload in zip(texts, vectors, payloads):
        payload["text"] = text
        payload.setdefault("scraped_at", now)
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=payload,
            )
        )

    client.upsert(collection_name=collection, points=points)
    print(f"[qdrant] Upserted {len(points)} points → {collection}")
    return len(points)


# ── Search ────────────────────────────────────────────────────────────────────

def search(
    collection: str,
    query_vector: list[float],
    top_k: int = 5,
    score_threshold: float = 0.5,
    filters: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """
    Semantic search in a collection.

    Args:
        collection:      Target collection name.
        query_vector:    Embedded query vector.
        top_k:           Max results to return.
        score_threshold: Minimum cosine similarity.
        filters:         Optional exact-match payload filters, e.g. {"platform": "reddit"}.

    Returns:
        List of payload dicts with an added 'score' field.
    """
    qdrant_filter = None
    if filters:
        conditions = [
            FieldCondition(key=k, match=MatchValue(value=v))
            for k, v in filters.items()
        ]
        qdrant_filter = Filter(must=conditions)

    results = client.search(
        collection_name=collection,
        query_vector=query_vector,
        limit=top_k,
        score_threshold=score_threshold,
        query_filter=qdrant_filter,
        with_payload=True,
    )

    return [{"score": r.score, **r.payload} for r in results]


def search_all_collections(
    query_vector: list[float],
    top_k_per_collection: int = 3,
    score_threshold: float = 0.5,
) -> dict[str, list[dict[str, Any]]]:
    """
    Search all 4 knowledge collections simultaneously.
    Returns a dict keyed by collection name.
    """
    return {
        name: search(name, query_vector, top_k_per_collection, score_threshold)
        for name in COLLECTIONS
    }


# ── Stats ─────────────────────────────────────────────────────────────────────

def collection_info() -> dict[str, int]:
    """Return point counts per collection."""
    info = {}
    for name in COLLECTIONS:
        try:
            count = client.count(collection_name=name).count
        except Exception:
            count = -1
        info[name] = count
    return info


if __name__ == "__main__":
    init_collections()
    print("[qdrant] Collections status:", collection_info())
