/**
 * scrapers/reddit.js
 */

import { shouldScrape, markScraped, insertRawChunks } from '../utils/cache.js';

async function extractRedditSentiment() {
  const sourceId = 'reddit.clashroyale_strategy';
  if (!shouldScrape(sourceId)) return;
  
  console.log(`\n--- Extracting Community Sentiment (Reddit) ---`);
  const url = 'https://www.reddit.com/r/ClashRoyale/search.json?q=flair_name%3A%22Strategy%22&restrict_sr=1&sort=hot';
  
  try {
    const response = await fetch(url, {
      headers: { 'User-Agent': 'ClashCoach_Prototype/1.0' } // Required to avoid instant blocks
    });
    
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    
    const rawData = await response.json();
    
    // Extract just the titles and the text bodies
    const posts = rawData.data.children.map(post => {
      return `Title: ${post.data.title}\nContent: ${post.data.selftext}`;
    });

    if (posts.length > 0) {
      console.log(`[reddit] Extracted Reddit Posts (Preview):`, posts[0].substring(0, 150).replace(/\n/g, ' '));
      
      // Save chunks to DB
      const payloads = posts.map(p => ({ source: 'reddit' }));
      const count = insertRawChunks('community_meta', posts, payloads);
      markScraped(sourceId, { count, ttlHours: 48 });
    } else {
       markScraped(sourceId, { count: 0, ttlHours: 48 });
    }

  } catch (error) {
    console.error(`[reddit] Fetch failed:`, error.message);
    markScraped(sourceId, { status: 'error', error: error.message });
  }
}

export default extractRedditSentiment;

if (process.argv[1].endsWith('reddit.js')) extractRedditSentiment();
