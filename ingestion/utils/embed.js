/**
 * utils/embed.js
 *
 * Mistral embedding utility.
 * Batches text chunks and returns 1024-dim vectors.
 * Used by all scrapers before upserting to Qdrant.
 */

import 'dotenv/config';

const MISTRAL_API_KEY = process.env.MISTRAL_API_KEY;
const EMBED_MODEL     = 'mistral-embed';
const BATCH_SIZE      = 64;   // max safe batch for Mistral
const RATE_LIMIT_MS   = 1000; // 1s pause between batches

/**
 * Embed an array of text strings using Mistral.
 * Automatically batches and rate-limits.
 *
 * @param {string[]} texts
 * @returns {Promise<number[][]>} Array of 1024-dim vectors
 */
export async function getEmbeddings(texts) {
  if (!texts.length) return [];

  if (!MISTRAL_API_KEY) {
    throw new Error('MISTRAL_API_KEY is not set in .env');
  }

  const allVectors = [];

  for (let i = 0; i < texts.length; i += BATCH_SIZE) {
    const batch = texts.slice(i, i + BATCH_SIZE);

    const res = await fetch('https://api.mistral.ai/v1/embeddings', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${MISTRAL_API_KEY}`,
        'Content-Type':  'application/json',
      },
      body: JSON.stringify({ model: EMBED_MODEL, input: batch }),
    });

    if (!res.ok) {
      const err = await res.text();
      throw new Error(`Mistral embed error ${res.status}: ${err}`);
    }

    const data = await res.json();
    const vectors = data.data.map(item => item.embedding);
    allVectors.push(...vectors);

    console.log(`[embed] Batch ${Math.floor(i / BATCH_SIZE) + 1}: embedded ${batch.length} chunks (total: ${allVectors.length})`);

    // Rate limit pause between batches
    if (i + BATCH_SIZE < texts.length) {
      await new Promise(r => setTimeout(r, RATE_LIMIT_MS));
    }
  }

  return allVectors;
}

/**
 * Split a long string into overlapping word-level chunks.
 *
 * @param {string} text
 * @param {number} chunkSize  Max words per chunk
 * @param {number} overlap    Words shared between adjacent chunks
 * @returns {string[]}
 */
export function chunkText(text, chunkSize = 300, overlap = 50) {
  const words = text.split(/\s+/).filter(Boolean);
  if (!words.length) return [];

  const chunks = [];
  let start = 0;

  while (start < words.length) {
    const end = Math.min(start + chunkSize, words.length);
    chunks.push(words.slice(start, end).join(' '));
    if (end === words.length) break;
    start += chunkSize - overlap;
  }

  return chunks;
}

/**
 * Embed chunks and return { texts, vectors } ready for Qdrant upsert.
 */
export async function embedChunks(texts) {
  const vectors = await getEmbeddings(texts);
  return { texts, vectors };
}
