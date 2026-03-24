# Takes the user command and extracts useful parameters

import re
def parse_command(text):
    text_lower = text.lower()
    params = {
        "size": None,       # in mm
        "motion": None,
        "speed": None,
        "direction": None
    }

    # --- Size (normalize everything to mm) ---
    size_match = re.search(r'(\d+(?:\.\d+)?)\s*(mm|cm|m|px)?', text_lower)
    if size_match:
        value = float(size_match.group(1))
        unit = size_match.group(2)
        unit_to_mm = {"mm": 1, "cm": 10, "m": 1000, "px": 0.264583}
        params["size"] = value * unit_to_mm[unit]

    # --- Motion ---
    motions = ["move", "translate", "rotate", "spin", "scale", "shift", "shrink", "expand", "grow"]
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

