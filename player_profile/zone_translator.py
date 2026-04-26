"""
player_profile/zone_translator.py

Translates hard mathematical coordinates (From YOLO bounding boxes or OpenCV template matches)
into English string concepts that the LLM Brain can understand without hallucinating.
"""

def get_zone_from_coords(x: int, y: int, screen_width: int = 400, screen_height: int = 800) -> str:
    """
    Given an X/Y radar ping from computer vision, returns the English lane and side.
    e.g. Y < 400 is Opponent side.
    """
    half_y = screen_height // 2
    half_x = screen_width // 2
    
    # 1. Determine Ownership (Side of the River)
    if y < half_y:
        side = "Opponent's Side"
    else:
        side = "User's Side"
        
    # 2. Determine Lane
    if x < half_x:
        lane = "Left Lane"
    else:
        lane = "Right Lane"
        
    # 3. Special Critical Zones (Princess Towers)
    # If it's very deep in the opponent's corners or user's corners:
    if y > screen_height * 0.75:
        if lane == "Left Lane":
            return "Left Princess Tower"
        if lane == "Right Lane":
            return "Right Princess Tower"
            
    if y < screen_height * 0.25:
        if lane == "Left Lane":
            return "Opponent's Left Princess Tower"
        if lane == "Right Lane":
            return "Opponent's Right Princess Tower"
            
    # If it's highly central near the river
    if (half_y - 50) < y < (half_y + 50):
        return f"the Bridge ({lane})"

    return f"the {side} ({lane})"

# Example Radar Translation
if __name__ == "__main__":
    test_ping = (320, 650) # Deep on the bottom right
    print(f"Radar Ping {test_ping} translates to: '{get_zone_from_coords(*test_ping)}'")
