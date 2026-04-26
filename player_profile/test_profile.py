import json
from clash_api import get_player, get_battlelog
from feature_extractor import extract_features
from gameplay_event_extractor import extract_gameplay_features
from profile_builder import build_ai_profile


PLAYER_TAG = "#VR08VQLL2"


player = get_player(PLAYER_TAG)
battlelog = get_battlelog(PLAYER_TAG)

api_features = extract_features(player, battlelog)

try:
    with open("video_events.json", "r") as f:
        gameplay_events = json.load(f)
except:
    print("Warning: video_events.json invalid or empty")
    gameplay_events = []

gameplay_features = extract_gameplay_features(gameplay_events)

ai_profile = build_ai_profile(api_features, gameplay_features)

with open("ai_player_profile.json", "w") as f:
    json.dump(ai_profile, f, indent=2)

print("Saved AI player profile to ai_player_profile.json")
print(json.dumps(ai_profile, indent=2))