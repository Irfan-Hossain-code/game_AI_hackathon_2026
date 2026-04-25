/**
 * utils/qdrant.js
 *
 * Thin wrapper around Qdrant's REST API.
 * No SDK needed — plain fetch calls.
 * Works with both in-memory (localhost:6333) and remote Qdrant instances.
 *
 * Collections managed:
 *   pro_tactics      — YouTube transcripts
 *   community_meta   — Reddit / Twitter posts
 *   patch_knowledge  — Supercell balance changes
 */

import 'dotenv/config';
import { randomUUID } from 'crypto';

const QDRANT_HOST = process.env.QDRANT_HOST || 'localhost';
const QDRANT_PORT = process.env.QDRANT_PORT || '6333';
const BASE_URL    = `http://${QDRANT_HOST}:${QDRANT_PORT}`;
const VECTOR_DIM  = 1024; // Mistral embed-large

const COLLECTIONS = ['pro_tactics', 'community_meta', 'patch_knowledge'];

// ── Internal fetch helper ─────────────────────────────────────────────────────

async function qdrantFetch(path, method = 'GET', body = null) {
  const opts = {
    method,
    headers: { 'Content-Type': 'application/json' },
  };
  if (body) opts.body = JSON.stringify(body);

  const res = await fetch(`${BASE_URL}${path}`, opts);

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Qdrant ${method} ${path} → ${res.status}: ${text}`);
  }

  return res.json();
}

// ── Init ──────────────────────────────────────────────────────────────────────

export async function initCollections() {
  const { result } = await qdrantFetch('/collections');
  const existing = new Set((result?.collections || []).map(c => c.name));

  for (const name of COLLECTIONS) {
    if (existing.has(name)) {
      console.log(`[qdrant] Collection exists: ${name}`);
      continue;
    }
    await qdrantFetch(`/collections/${name}`, 'PUT', {
      vectors: { size: VECTOR_DIM, distance: 'Cosine' },
    });
    console.log(`[qdrant] Created collection: ${name}`);
  }
}

// ── Upsert ────────────────────────────────────────────────────────────────────

/**
 * Insert points into a Qdrant collection.
 *
 * @param {string}   collection  Target collection name
 * @param {string[]} texts       Raw text per point (stored in payload)
 * @param {number[][]} vectors   Pre-computed embeddings
 * @param {object[]} payloads    Metadata per point
 */
export async function upsertPoints(collection, texts, vectors, payloads) {
  if (!texts.length) return 0;

  const now = new Date().toISOString();
  const points = texts.map((text, i) => ({
    id:      randomUUID(),
    vector:  vectors[i],
    payload: { text, scraped_at: now, ...payloads[i] },
  }));

  await qdrantFetch(`/collections/${collection}/points`, 'PUT', { points });
  console.log(`[qdrant] Upserted ${points.length} points → ${collection}`);
  return points.length;
}

// ── Search ────────────────────────────────────────────────────────────────────

export async function search(collection, queryVector, { topK = 5, scoreThreshold = 0.5 } = {}) {
  const { result } = await qdrantFetch(`/collections/${collection}/points/search`, 'POST', {
    vector:           queryVector,
    limit:            topK,
    score_threshold:  scoreThreshold,
    with_payload:     true,
  });
  return (result || []).map(r => ({ score: r.score, ...r.payload }));
}

// ── Stats ─────────────────────────────────────────────────────────────────────

export async function collectionStats() {
  const stats = {};
  for (const name of COLLECTIONS) {
    try {
      const { result } = await qdrantFetch(`/collections/${name}`);
      stats[name] = result?.vectors_count ?? 0;
    } catch {
      stats[name] = -1;
    }
  }
  return stats;
}
