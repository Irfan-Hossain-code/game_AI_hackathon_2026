from clash_api import get_player, get_battlelog
from feature_extractor import extract_features
from profile_rules import build_profile


def build_player_profile(player_tag):
    player = get_player(player_tag)
    battlelog = get_battlelog(player_tag)

    features = extract_features(player, battlelog)
    profile = build_profile(features)

    return profile