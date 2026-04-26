/**
 * scrapers/youtube.js
 */

import { YoutubeTranscript } from 'youtube-transcript';
import { shouldScrape, markScraped, insertRawChunks } from '../utils/cache.js';

const SEED_VIDEOS = ['dQw4w9WgXcQ']; // Example: Ian77 or CRL

async function extractYouTubeTactics() {
  const sourceId = 'youtube.pro_transcripts';
  if (!shouldScrape(sourceId)) return;
  
  console.log(`\n--- Extracting Pro Tactics (YouTube) ---`);
  let totalChunks = 0;
  
  for (const videoId of SEED_VIDEOS) {
    try {
      const transcript = await YoutubeTranscript.fetchTranscript(videoId);
      
      // Combine the array of text chunks into one massive string for the LLM
      const fullText = transcript.map(t => t.text).join(' ');
      
      console.log(`[youtube] Extracted Tactics:`, fullText.substring(0, 150) + "...");
      
      // Save it into sqlite for the Python AI layer to embed later
      insertRawChunks('pro_tactics', [fullText], [{ source: 'youtube', video_id: videoId }]);
      totalChunks++;
      
    } catch (error) {
      console.error(`[youtube] Transcript failed for ${videoId}:`, error.message);
    }
  }

  markScraped(sourceId, { count: totalChunks, ttlHours: 72 });
}

export default extractYouTubeTactics;

if (process.argv[1].endsWith('youtube.js')) extractYouTubeTactics();
