/**
 * scrapers/patch_notes.js
 */

import * as cheerio from 'cheerio';
import { shouldScrape, markScraped, insertRawChunks } from '../utils/cache.js';

async function extractPatchNotes() {
  const sourceId = 'supercell.patch_notes';
  if (!shouldScrape(sourceId)) return;
  
  console.log(`\n--- Extracting the Rulebook (Patch Notes) ---`);
  
  // They sometimes shift between /blog/release-notes and /blog/balance-changes.
  // Using the exact URL provided first.
  const url = 'https://supercell.com/en/games/clashroyale/blog/release-notes/';
  
  try {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    
    const html = await response.text();
    const $ = cheerio.load(html);
    
    let textContent = '';
    // Target article paragraphs
    $('article p, article h2, article h3, .blog-post p').each((i, element) => {
      textContent += $(element).text() + '\n';
    });

    console.log(`[patch_notes] Extracted Patch Notes (Preview):`, textContent.substring(0, 150).replace(/\n/g, ' '));
    
    if (textContent.length > 50) {
      insertRawChunks('patch_knowledge', [textContent], [{ source: 'supercell_blog', url }]);
      markScraped(sourceId, { count: 1, ttlHours: 168 });
    } else {
      markScraped(sourceId, { count: 0, ttlHours: 168 });
    }
  } catch (error) {
    console.error(`[patch_notes] Extraction failed:`, error.message);
    markScraped(sourceId, { status: 'error', error: error.message });
  }
}

export default extractPatchNotes;

if (process.argv[1].endsWith('patch_notes.js')) extractPatchNotes();
