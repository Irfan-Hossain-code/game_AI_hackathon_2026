import json
from profile_service import build_player_profile
from gameplay_event_extractor import get_mock_gameplay_events, extract_gameplay_features


PLAYER_TAG = "#VR08VQLL2"


# API-based profile
profile = build_player_profile(PLAYER_TAG)


# Gameplay-based profile (mock for now)
events = get_mock_gameplay_events()
gameplay_features = extract_gameplay_features(events)


# Combine both
combined_output = {
    "api_profile": profile,
    "gameplay_profile": gameplay_features
}


print(json.dumps(combined_output, indent=2))