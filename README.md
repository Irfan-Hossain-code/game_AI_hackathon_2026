# Clash Royale AI Coach - Backend & Knowledge Engine

This repository contains the full Node.js Web Scraper, SQLite Meta Database, Qdrant Vector Engine, and Mistral Coaching RAG Pipeline. 

This guide is specifically for setting up the **Brain** (Storage/RAG engine) independent of the Video Extractor.

## 1. Initial Setup

### Prerequisites
- Python 3.10+
- Node.js (v18+)
- Mistral API Key (For embedding and LLM generation)
- Clash Royale API Key (Optional: for live player lookups)

### Environment Variables
Create a `.env` file in the root directory and add your keys (see `.env.template`):
```bash
MISTRAL_API_KEY="your_key_here"
CLASH_API_TOKEN="your_key_here"
```

### Installation
1. **Initialize the Python Environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

2. **Initialize the Node Scraper:**
    ```bash
    cd ingestion
    npm install
    cd ..
    ```

---

## 2. Populating the Databases (SQLite & Qdrant)
Before the AI Coach can give advice, you must seed the Vector Engine with the meta data.

1. **Run the Web Scrapers (Node.js)**
    This will pull YouTube Transcripts, Reddit threads, and Meta Decklists into the local `knowledge.db` SQLite storage.
    ```bash
    cd ingestion
    npm run ingest
    cd ..
    ```

2. **Embed the Data (Python)**
    This pushes everything from SQLite into Qdrant using the Mistral Embedding framework so it can be searched semantically.
    ```bash
    .venv/bin/python coaching/embed_knowledge.py
    ```

---

## 3. Retrieving Coaching Output
The coaching pipeline is independent of the video platform. It exposes functions you can import directly into your own event-mapping script. 

### A. Dynamic RAG Query (Using your own Python script)
If you have your own script outputting game state JSON arrays, you can pipe it directly to our Mistral API loop:
```python
from coaching.game_loop import react_to_game_state

# Example state from your custom video parser
custom_tracker_output = {
    "timestamp": "0:45",
    "opponent_action": "Placed Goblin Barrel",
    "location": "Right Princess Tower",
    "user_hand": ["Hog Rider", "Log", "Ice Spirit", "Musketeer"],
    "user_elixir": 5
}

# Submits state -> Queries Qdrant -> Yields Mistral LLM prompt
react_to_game_state(custom_tracker_output)
```

### B. Live Player Profile Setup
If you want to pull live Supercell Cloud stats directly:
```bash
.venv/bin/python coaching/coach_service.py "#YOUR_PLAYER_TAG"
```
