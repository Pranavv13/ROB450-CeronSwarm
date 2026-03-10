from google import genai
import os, re, json, csv, ast
import text_parsing

# Initialize the client
# api = os.getenv("GEMINI_API_KEY")

def send_command(user_prompt, parameters):
    client = genai.Client(api_key = "AIzaSyDDcjJnD5W8OEu9N1vx178kKzMlHCDQyH0")
    
    gemini_prompt = ("For the following command, generated an " \
    "8x8 matrix of 1s and 0s representing the shape listed." \
    "1s are the shape and 0s are background." \
    "Ignore any size, movement, or speed parameters." \
    "Nothing else just the matrix. " + user_prompt
    )

    # print(f"Parsed params: {json.dumps(params, indent=2)}")

    try:
        print("Sending request to Gemini")
        response = client.models.generate_content(
            model="gemini-3-flash-preview", 
            contents=[gemini_prompt]
        )
        # print(response.text)
    
        # Extract matrix
        matrix = []

        for line in response.text.strip().split("\n"):
            row = [int(x) for x in line.split()]
            matrix.append(row)

        # Export to CSV
        filename = "shape_output.csv"

        with open(filename, "w", newline="") as file:
            writer = csv.writer(file)

            # Write matrix to CSV
            for row in matrix:
                writer.writerow(row)

            # Blank row separator
            writer.writerow([])

            # Write parameters to CSV
            writer.writerow(["size", parameters["size"]])
            writer.writerow(["movement", parameters["movement"]])
            writer.writerow(["speed", parameters["speed"]])
            writer.writerow(["direction", parameters["direction"]])

        print(f"Data exported to {filename}")

    except Exception as e:
        print(f"Connection failed: {e}")

    

# Single Run - No Loop for Now
user_input = input("Enter command: ").strip()
params = text_parsing.parse_command(user_input)
send_command(user_input, params)

