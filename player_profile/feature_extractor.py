def card_names(cards):
    return [card.get("name") for card in cards]


def average_elixir(cards):
    costs = [card.get("elixirCost") for card in cards if card.get("elixirCost") is not None]

    if not costs:
        return 0

    return round(sum(costs) / len(costs), 2)


def get_player_from_battle(battle, player_tag):
    for player in battle.get("team", []):
        if player.get("tag") == player_tag:
            return player

    return None


def get_opponent_from_battle(battle):
    opponents = battle.get("opponent", [])

    if not opponents:
        return None

    return opponents[0]


def extract_features(player, battlelog):
    player_tag = player.get("tag")

    recent_matches = []
    recent_wins = 0
    crowns_for = 0
    crowns_against = 0
    total_elixir_leaked = 0
    deck_history = []

    for battle in battlelog:
        me = get_player_from_battle(battle, player_tag)
        opponent = get_opponent_from_battle(battle)

        if me is None or opponent is None:
            continue

        my_crowns = me.get("crowns", 0)
        opponent_crowns = opponent.get("crowns", 0)

        won = my_crowns > opponent_crowns

        if won:
            recent_wins += 1

        crowns_for += my_crowns
        crowns_against += opponent_crowns
        total_elixir_leaked += me.get("elixirLeaked", 0)

        deck = card_names(me.get("cards", []))
        deck_history.append(deck)

        recent_matches.append({
            "battle_time": battle.get("battleTime"),
            "game_mode": battle.get("gameMode", {}).get("name"),
            "won": won,
            "crowns_for": my_crowns,
            "crowns_against": opponent_crowns,
            "elixir_leaked": me.get("elixirLeaked", 0),
            "deck": deck,
            "opponent_deck": card_names(opponent.get("cards", []))
        })

    match_count = len(recent_matches)

    current_deck = player.get("currentDeck", [])

    features = {
        "player_tag": player.get("tag"),
        "player_name": player.get("name"),
        "trophies": player.get("trophies"),
        "best_trophies": player.get("bestTrophies"),
        "arena": player.get("arena", {}).get("name"),
        "battle_count": player.get("battleCount", 0),
        "wins": player.get("wins", 0),
        "losses": player.get("losses", 0),
        "overall_win_rate": safe_divide(player.get("wins", 0), player.get("wins", 0) + player.get("losses", 0)),
        "three_crown_rate": safe_divide(player.get("threeCrownWins", 0), player.get("battleCount", 0)),
        "favorite_card": player.get("currentFavouriteCard", {}).get("name"),
        "current_deck": card_names(current_deck),
        "current_deck_average_elixir": average_elixir(current_deck),
        "recent_match_count": match_count,
        "recent_win_rate": safe_divide(recent_wins, match_count),
        "average_crowns_for": safe_divide(crowns_for, match_count),
        "average_crowns_against": safe_divide(crowns_against, match_count),
        "average_elixir_leaked": safe_divide(total_elixir_leaked, match_count),
        "recent_matches": recent_matches
    }

    return features


def safe_divide(a, b):
    if b == 0:
        return 0

    return round(a / b, 2)