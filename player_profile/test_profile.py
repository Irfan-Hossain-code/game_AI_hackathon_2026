import json
from profile_service import build_player_profile


PLAYER_TAG = "#VR08VQLL2"


profile = build_player_profile(PLAYER_TAG)

print(json.dumps(profile, indent=2))