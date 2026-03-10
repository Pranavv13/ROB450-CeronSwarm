from google import genai
import re, json

client = genai.Client(api_key="AIzaSyDMqSfAUWfOQkxcyTda43WabKXPiGNx3SM")

def parse_command(text):
    text_lower = text.lower()
    params = {
    #     "shape": None,
        "size": None,       # in mm
        "movement": None,
        "speed": None,
        "direction": None
    }

    # --- Shape ---
    # for shape in ["circle", "square", "triangle", "rectangle", "line", "ellipse", "hexagon"]:
    #     if shape in text_lower:
    #         params["shape"] = shape
    #         break

    # --- Size (normalize everything to mm) ---
    size_match = re.search(r'(\d+(?:\.\d+)?)\s*(mm|cm|m|px)?', text_lower)
    if size_match:
        value = float(size_match.group(1))
        unit = size_match.group(2) or "mm"
        unit_to_mm = {"mm": 1, "cm": 10, "m": 1000, "px": 0.264583}
        params["size"] = value * unit_to_mm[unit]

    # --- Movement ---
    for move in ["move", "translate", "rotate", "spin", "scale", "shift", "shrink", "expand", "grow"]:
        if move in text_lower:
            params["movement"] = move
            break

    # --- Speed ---
    speed_match = re.search(r'(\d+(?:\.\d+)?)\s*(mm/s|cm/s|m/s)', text_lower)
    if speed_match:
        value = float(speed_match.group(1))
        unit = speed_match.group(2)
        unit_to_mms = {"mm/s": 1, "cm/s": 10, "m/s": 1000}
        params["speed"] = value * unit_to_mms[unit]  # stored as mm/s
    else:
        for word, val in [("slowly", 5), ("slow", 5), ("fast", 50), ("quickly", 50), ("rapid", 100)]:
            if word in text_lower:
                params["speed"] = val
                break

    # --- Direction ---
    for direction in ["left", "right", "up", "down", "forward", "backward", "clockwise", "counterclockwise"]:
        if direction in text_lower:
            params["direction"] = direction
            break

    return params


# --- Main loop ---
while True:
    user_input = input("\nEnter command (or 'quit'): ").strip()
    if user_input.lower() == "quit":
        break

    params = parse_command(user_input)
    print(f"Parsed params: {json.dumps(params, indent=2)}")

    # shape = params["shape"]
    # if not shape:
    #     print("No shape detected in command, please include a shape (e.g. circle, square...)")
    #     continue

    # Build prompt
    # size_hint = f" of size {params['size']}mm" if params["size"] else ""
    # prompt = (
    #     f"Generate an 8x8 matrix of 1s and 0s representing a {shape}{size_hint}. "
    #     f"0s are background and 1s are the {shape}. "
    #     f"Nothing else, just the matrix."
    # )

    # try:
    #     print("Sending request...")
    #     response = client.models.generate_content(
    #         model="gemini-2.0-flash",
    #         contents=[prompt]
    #     )
    #     print(f"\nMatrix:\n{response.text}")
    #     print(f"\nOther params — movement: {params['movement']}, "
    #           f"speed: {params['speed']} mm/s, direction: {params['direction']}")

    # except Exception as e:
    #     print(f"Request failed: {e}")