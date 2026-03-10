from google import genai
import os

# Initialize the client
# api = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key = "YOUR API KEYT")
    # http_options=types.HttpOptions(api_version="v1")
#)

shape = "circle"
prompt = "Generate a 8x8 matrix of 1s and 0s representing a " + shape + ". " \
"0s are background and 1s are the" + shape + ". " \
"Nothing else just the matrix."
try:
    print("Sending request...")
    response = client.models.generate_content(
        model="gemini-3-flash-preview", 
        contents=[prompt]
    )
    print(f"{response.text}")
    

except Exception as e:
    print(f"Connection failed: {e}")

# TO DO 
# Read shape from user prompt to terminal not Python script 
# Export reponse.text (the matrix) to a CSV File 
# Multiple prompts in one script 

# Later TO DO 
# Sizes and movement/velocity 