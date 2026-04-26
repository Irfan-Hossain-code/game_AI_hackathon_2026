/**
 * run_all.js
 *
 * Orchestrates the Node.js ingestion pipeline.
 */

import scrapeYouTube from './scrapers/youtube.js';
import scrapeReddit from './scrapers/reddit.js';
import scrapePatchNotes from './scrapers/patch_notes.js';
import scrapeDeckshopMeta from './scrapers/deckshop.js';
import { getAllLogs } from './utils/cache.js';

async function runAll() {
  console.log('=============================================');
  console.log('   CLASH ROYALE INGESTION PIPELINE (NODE)    ');
  console.log('=============================================');

  await scrapeDeckshopMeta();
  await scrapePatchNotes();
  await scrapeReddit();
  await scrapeYouTube();

  console.log('\n=============================================');
  console.log('   PIPELINE COMPLETE - CACHE STATUS          ');
  console.log('=============================================');
  
  const logs = getAllLogs();
  for (const log of logs) {
    const ageHours = (Date.now() - new Date(log.last_scraped).getTime()) / 3600000;
    const isFresh = ageHours < log.ttl_hours;
    console.log(`${log.source.padEnd(30)} | ${isFresh ? 'FRESH' : 'STALE'} | ${log.record_count} items`);
  }
}

const args = process.argv.slice(2);
if (args.includes('--status')) {
  console.log('\n--- CACHE STATUS ---');
  const logs = getAllLogs();
  if (!logs || logs.length === 0) {
    console.log("Cache is empty. Nothing has been scraped yet.");
  } else {
    for (const log of logs) {
      console.log(`${log.source.padEnd(30)} | Updated: ${log.last_scraped} | Rows: ${log.record_count}`);
    }
  }
} else if (args.includes('--force')) {
  console.log("Force update not fully implemented (requires deleting sqlite cache entries manually or via func). Running normally...");
  runAll().catch(console.error);
} else {
  runAll().catch(console.error);
}
