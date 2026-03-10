from google import genai
import os, re, json
import text_parsing

# Initialize the client
# api = os.getenv("GEMINI_API_KEY")

def send_command(parameters):
    client = genai.Client(api_key = "AIzaSyDDcjJnD5W8OEu9N1vx178kKzMlHCDQyH0")
    
    prompt = "Generate a 8x8 matrix of 1s and 0s \
          representing a " + parameters + ". " \
    "0s are background and 1s are the" + parameters + ". " \
    "Nothing else just the matrix." \
    "Ignore any size, movement, or direction requests. Just make the shape"

    print(f"Parsed params: {json.dumps(params, indent=2)}")

    try:
        print("Sending request to Gemini")
        response = client.models.generate_content(
            model="gemini-3-flash-preview", 
            contents=[prompt]
        )
        print(f"{response.text}")
    

    except Exception as e:
        print(f"Connection failed: {e}")

# --- Main loop ---
while True:
    user_input = input("Enter command: ").strip()
    params = text_parsing.parse_command(user_input)
    send_command(user_input)
