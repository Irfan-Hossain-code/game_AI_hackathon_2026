/**
 * utils/cache.js
 *
 * SQLite-backed TTL cache for all scrapers.
 * DB file: ../knowledge.db
 */

import Database from 'better-sqlite3';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DB_PATH = path.resolve(__dirname, '../../knowledge.db');

let _db = null;

function getDb() {
  if (!_db) {
    _db = new Database(DB_PATH);
    _db.pragma('journal_mode = WAL');
    initSchema();
  }
  return _db;
}

function initSchema() {
  getDb().exec(`
    CREATE TABLE IF NOT EXISTS scrape_log (
      source        TEXT PRIMARY KEY,
      last_scraped  TEXT,
      status        TEXT DEFAULT 'ok',
      record_count  INTEGER DEFAULT 0,
      ttl_hours     INTEGER DEFAULT 24,
      error_message TEXT
    );

    CREATE TABLE IF NOT EXISTS meta_decks (
      id            INTEGER PRIMARY KEY AUTOINCREMENT,
      deck_hash     TEXT UNIQUE NOT NULL,
      cards_json    TEXT NOT NULL,
      usage_rate    REAL DEFAULT 0,
      win_rate      REAL DEFAULT 0,
      season        TEXT,
      source        TEXT,
      scraped_at    TEXT
    );

    -- NEW: Store text chunks for later Python embedding
    CREATE TABLE IF NOT EXISTS raw_chunks (
      id            INTEGER PRIMARY KEY AUTOINCREMENT,
      collection    TEXT NOT NULL,
      text          TEXT NOT NULL,
      payload_json  TEXT NOT NULL,
      scraped_at    TEXT NOT NULL,
      embedded      INTEGER DEFAULT 0
    );
  `);
}

export function shouldScrape(source) {
  initSchema();
  const row = getDb().prepare('SELECT * FROM scrape_log WHERE source = ?').get(source);

  if (!row || !row.last_scraped || row.status === 'error') return true;
  const ageHours = (Date.now() - new Date(row.last_scraped).getTime()) / 3_600_000;
  return ageHours >= row.ttl_hours;
}

export function markScraped(source, { count = 0, ttlHours = 24, status = 'ok', error = null } = {}) {
  initSchema();
  getDb().prepare(`
    INSERT INTO scrape_log (source, last_scraped, status, record_count, ttl_hours, error_message)
    VALUES (?, ?, ?, ?, ?, ?)
    ON CONFLICT(source) DO UPDATE SET
      last_scraped  = excluded.last_scraped,
      status        = excluded.status,
      record_count  = excluded.record_count,
      ttl_hours     = excluded.ttl_hours,
      error_message = excluded.error_message
  `).run(source, new Date().toISOString(), status, count, ttlHours, error);
}

export function getAllLogs() {
  initSchema();
  return getDb().prepare('SELECT * FROM scrape_log ORDER BY source').all();
}

export function upsertMetaDecks(decks) {
  initSchema();
  const stmt = getDb().prepare(`
    INSERT INTO meta_decks (deck_hash, cards_json, usage_rate, win_rate, season, source, scraped_at)
    VALUES (@deck_hash, @cards_json, @usage_rate, @win_rate, @season, @source, @scraped_at)
    ON CONFLICT(deck_hash) DO UPDATE SET
      usage_rate = excluded.usage_rate,
      win_rate   = excluded.win_rate,
      scraped_at = excluded.scraped_at
  `);
  const now = new Date().toISOString();
  const insertMany = getDb().transaction((rows) => {
    for (const row of rows) stmt.run({ scraped_at: now, ...row });
  });
  insertMany(decks);
  return decks.length;
}

export function insertRawChunks(collection, texts, payloads) {
  initSchema();
  const stmt = getDb().prepare(`
    INSERT INTO raw_chunks (collection, text, payload_json, scraped_at)
    VALUES (@collection, @text, @payload_json, @scraped_at)
  `);
  const now = new Date().toISOString();
  const insertMany = getDb().transaction(() => {
    for (let i = 0; i < texts.length; i++) {
       stmt.run({
         collection,
         text: texts[i],
         payload_json: JSON.stringify(payloads[i]),
         scraped_at: now
       });
    }
  });
  insertMany();
  return texts.length;
}
