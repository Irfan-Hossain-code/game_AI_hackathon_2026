def get_mock_gameplay_events():
    return [
        {
            "timestamp": 4.2,
            "card": "Giant",
            "location": {"lane": "left", "side": "own_half", "x": 0.32, "y": 0.78},
            "context": "opening_play",
            "confidence": 0.85,
            "source": "mock"
        },
        {
            "timestamp": 12.8,
            "card": "Musketeer",
            "location": {"lane": "left", "side": "own_half", "x": 0.38, "y": 0.88},
            "context": "supporting_push",
            "confidence": 0.8,
            "source": "mock"
        },
        {
            "timestamp": 25.0,
            "card": "Fireball",
            "location": {"lane": "right", "side": "enemy_half", "x": 0.68, "y": 0.35},
            "context": "spell_pressure",
            "confidence": 0.75,
            "source": "mock"
        }
    ]


def extract_gameplay_features(events):
    evidence = []

    if not events:
        return {
            "event_count": 0,
            "opening_style": "unknown",
            "pressure_style": "unknown",
            "cards_played_sequence": [],
            "evidence": ["No gameplay events detected yet."],
            "raw_events": []
        }

    first_event = events[0]
    first_card_time = first_event["timestamp"]

    cards_played = []
    lane_counts = {"left": 0, "right": 0, "middle": 0, "unknown": 0}
    own_half_plays = 0
    enemy_half_plays = 0

    for event in events:
        cards_played.append(event["card"])

        lane = event.get("location", {}).get("lane", "unknown")
        lane_counts[lane] = lane_counts.get(lane, 0) + 1

        side = event.get("location", {}).get("side", "unknown")

        if side == "own_half":
            own_half_plays += 1
        elif side == "enemy_half":
            enemy_half_plays += 1

    if first_card_time < 10:
        opening_style = "early proactive play"
        evidence.append(
            f"Player opened early with {first_event['card']} at {first_card_time}s."
        )
    else:
        opening_style = "patient opening"
        evidence.append(
            f"Player waited until {first_card_time}s before first detected play."
        )

    if enemy_half_plays > own_half_plays:
        pressure_style = "forward pressure"
        evidence.append(
            f"Most detected plays were in enemy half: {enemy_half_plays}/{len(events)}."
        )
    elif own_half_plays > enemy_half_plays:
        pressure_style = "defensive buildup"
        evidence.append(
            f"Most detected plays were in own half: {own_half_plays}/{len(events)}."
        )
    else:
        pressure_style = "balanced placement"
        evidence.append("Detected plays were balanced between own half and enemy half.")

    most_used_lane = max(lane_counts, key=lane_counts.get)
    evidence.append(f"Most used lane was {most_used_lane} lane.")

    return {
        "event_count": len(events),
        "opening_play": first_event,
        "first_card_time": first_card_time,
        "cards_played_sequence": cards_played,
        "lane_counts": lane_counts,
        "own_half_plays": own_half_plays,
        "enemy_half_plays": enemy_half_plays,
        "opening_style": opening_style,
        "pressure_style": pressure_style,
        "evidence": evidence,
        "raw_events": events
    }