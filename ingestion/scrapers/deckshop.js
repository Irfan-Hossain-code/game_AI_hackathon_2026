/**
 * scrapers/deckshop.js
 */

import * as cheerio from 'cheerio';
import crypto from 'crypto';
import { shouldScrape, markScraped, upsertMetaDecks } from '../utils/cache.js';

function hashDeck(cards) {
  return crypto.createHash('md5').update([...cards].sort().join('|')).digest('hex');
}

async function extractMetaDecks() {
  const sourceId = 'deckshop.meta_decks';
  if (!shouldScrape(sourceId)) return;
  
  console.log(`\n--- Extracting Meta Stats (Deckshop) ---`);
  const url = 'https://www.deckshop.pro/deck/list/meta';
  
  try {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    
    const html = await response.text();
    const $ = cheerio.load(html);
    
    const decksData = [];
    $('.deck-layout').each((i, el) => {
      const cards = [];
      $(el).find('img.card-img').each((j, img) => {
        cards.push($(img).attr('alt').trim());
      });
      if (cards.length === 8) {
        decksData.push({
          deck_hash: hashDeck(cards),
          cards_json: JSON.stringify(cards),
          usage_rate: 0.0,
          win_rate: 0.0,
          season: 'current',
          source: 'deckshop_html'
        });
      }
    });

    if (decksData.length > 0) {
      console.log(`[deckshop] Extracted Meta Decks:`, decksData[0].cards_json);
      const count = upsertMetaDecks(decksData);
      markScraped(sourceId, { count, ttlHours: 24 });
    } else {
       console.log(`[deckshop] No decks found.`);
       markScraped(sourceId, { count: 0, ttlHours: 24 });
    }

  } catch (error) {
    console.error(`[deckshop] Deck extraction failed:`, error.message);
    markScraped(sourceId, { status: 'error', error: error.message });
  }
}

export default extractMetaDecks;

if (process.argv[1].endsWith('deckshop.js')) extractMetaDecks();
