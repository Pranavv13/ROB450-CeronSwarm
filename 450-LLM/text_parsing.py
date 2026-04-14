# Takes the user command and extracts useful parameters

import re, json

# Ordered from longest to shortest to avoid partial matches (e.g. "star" before "star fish")
_KNOWN_SHAPES = [
    "triangle", "rectangle", "pentagon", "hexagon", "octagon",
    "diamond", "ellipse", "circle", "square", "cross", "arrow",
    "heart", "star", "ring", "oval", "line", "spiral",
]

def extract_shape_name(text):
    """
    Return the first recognised shape word in text, or None.
    Also catches patterns like 'make a <word>', 'form a <word>', etc.
    so uncommon shapes (e.g. 'horseshoe') are still captured.
    """
    text_lower = text.lower()

    # Fast path: check against known shape list first
    for shape in _KNOWN_SHAPES:
        if shape in text_lower:
            return shape

    # Fallback: extract the noun right after an action verb
    action_pattern = r'(?:make|form|create|draw|build|generate|shape)\s+(?:a\s+|an\s+)?([a-z]+)'
    match = re.search(action_pattern, text_lower)
    if match:
        return match.group(1)

    return None


def parse_command(text):
    text_lower = text.lower()
    params = {
        "shape": extract_shape_name(text),
        "size": None,       # in mm
        "motion": None,
        "speed": None,
        "direction": None,
    }

    # --- Size (normalize everything to mm) ---
    size_match = re.search(r'(\d+(?:\.\d+)?)\s*(mm|cm|m|px)', text_lower)
    if size_match:
        value = float(size_match.group(1))
        unit = size_match.group(2)
        unit_to_mm = {"mm": 1, "cm": 10, "m": 1000, "px": 0.264583}
        params["size"] = value * unit_to_mm[unit]

    # --- Motion ---
    motions = ["move", "translate", "shift"]
    for m in motions:
        if m in text_lower:
            params["motion"] = m
            break

    # --- Speed ---
    speed_match = re.search(r'(\d+(?:\.\d+)?)\s*(mm/s|cm/s|m/s)', text_lower)
    speeds = [("slowly", 5), ("slow", 5), ("fast", 50), ("quickly", 50), ("rapid", 100)]
    if speed_match:
        value = float(speed_match.group(1))
        unit = speed_match.group(2)
        unit_to_mms = {"mm/s": 1, "cm/s": 10, "m/s": 1000}
        params["speed"] = value * unit_to_mms[unit]  # stored as mm/s
    else:
        for word, val in speeds:
            if word in text_lower:
                params["speed"] = val
                break

    # --- Direction ---
    directions = ["left", "right", "up", "down", "forward", "backward", "clockwise", "counterclockwise"]
    for d in directions:
        if d in text_lower:
            params["direction"] = d
            break

    return params


# --- Main loop (only runs when this file is executed directly) ---
if __name__ == "__main__":
    while True:
        user_input = input("\nEnter command (or 'quit'): ").strip()
        if user_input.lower() == "quit":
            break

        params = parse_command(user_input)
        print(f"Parsed params: {json.dumps(params, indent=2)}")
