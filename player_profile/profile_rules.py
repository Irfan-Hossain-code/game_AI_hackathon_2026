def build_profile(features):
    strengths = []
    weaknesses = []
    evidence = []

    avg_elixir = features["current_deck_average_elixir"]
    recent_win_rate = features["recent_win_rate"]
    three_crown_rate = features["three_crown_rate"]
    avg_elixir_leaked = features["average_elixir_leaked"]

    if avg_elixir <= 3.2:
        deck_style = "fast cycle"
        evidence.append(f"Current deck average elixir is low: {avg_elixir}")
    elif avg_elixir >= 4.0:
        deck_style = "heavy beatdown"
        evidence.append(f"Current deck average elixir is high: {avg_elixir}")
    else:
        deck_style = "balanced"
        evidence.append(f"Current deck average elixir is balanced: {avg_elixir}")

    if three_crown_rate >= 0.25:
        aggression = "high"
        strengths.append("Strong tower-finishing ability")
        evidence.append(f"Three-crown rate is {three_crown_rate}")
    elif three_crown_rate >= 0.10:
        aggression = "medium"
        evidence.append(f"Three-crown rate is moderate: {three_crown_rate}")
    else:
        aggression = "low"
        evidence.append(f"Three-crown rate is low: {three_crown_rate}")

    if avg_elixir_leaked <= 1.5:
        tempo = "efficient"
        strengths.append("Good elixir usage with low leaking")
        evidence.append(f"Average leaked elixir is low: {avg_elixir_leaked}")
    elif avg_elixir_leaked <= 4:
        tempo = "moderate"
        evidence.append(f"Average leaked elixir is moderate: {avg_elixir_leaked}")
    else:
        tempo = "passive"
        weaknesses.append("Leaks too much elixir")
        evidence.append(f"Average leaked elixir is high: {avg_elixir_leaked}")

    if recent_win_rate >= 0.6:
        strengths.append("Currently performing well in recent matches")
    elif recent_win_rate <= 0.4:
        weaknesses.append("Recent win rate is low")

    playstyle = f"{tempo} {deck_style} player with {aggression} aggression"

    return {
        "player_tag": features["player_tag"],
        "player_name": features["player_name"],
        "playstyle": playstyle,
        "deck_style": deck_style,
        "aggression": aggression,
        "tempo": tempo,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "evidence": evidence,
        "features": features
    }