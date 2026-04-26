def extract_gameplay_features(events):
    lane_counts = {}
    own_half_plays = 0
    enemy_half_plays = 0

    for event in events:
        lane = event.get("location", {}).get("lane", "unknown")
        side = event.get("location", {}).get("side", "unknown")

        lane_counts[lane] = lane_counts.get(lane, 0) + 1

        if side == "own_half":
            own_half_plays += 1
        elif side == "enemy_half":
            enemy_half_plays += 1

    return {
        "opening_style": "early proactive play" if events else "unknown",
        "pressure_style": "aggressive" if enemy_half_plays > own_half_plays else "balanced",
        "event_count": len(events),
        "own_half_plays": own_half_plays,
        "enemy_half_plays": enemy_half_plays,
        "lane_counts": lane_counts,
        "evidence": events[:6],
        "raw_events": events
    }
