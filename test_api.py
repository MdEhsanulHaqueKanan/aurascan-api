import requests
import json

# The URL of our running Flask API
# API_URL = "http://127.0.0.1:5000/analyze"
API_URL = "http://127.0.0.1:5000/analyze"

# The URL of our LIVE Hugging Face API
# API_URL = "https://ehsanulhaque92-aurascanai.hf.space/analyze"

# The path to our test image
IMAGE_PATH = "test_image.jpg"

def test_prediction():
    """
    Sends a test image to the running API and prints the response.
    """
    print(f"--- Sending test image to {API_URL} ---")
    
    try:
        # Open the image file in binary read mode
        with open(IMAGE_PATH, 'rb') as image_file:
            # The 'files' dictionary is how you send a multipart/form-data request
            files = {'file': (IMAGE_PATH, image_file, 'image/jpeg')}
            
            # Send the POST request
            response = requests.post(API_URL, files=files)
            
            # Check if the request was successful
            if response.status_code == 200:
                print("\n--- SUCCESS: Received a valid response ---")
                # Pretty-print the JSON response
                print(json.dumps(response.json(), indent=2))
            else:
                print(f"\n--- ERROR: Server returned status code {response.status_code} ---")
                print("Response body:")
                print(response.text)
    
    except FileNotFoundError:
        print(f"\n--- FATAL ERROR ---")
        print(f"Test image not found at '{IMAGE_PATH}'.")
        print("Please place a car image in the root of your project and name it 'test_image.jpg'.")
    except requests.exceptions.ConnectionError:
        print(f"\n--- FATAL ERROR ---")
        print(f"Could not connect to the server at {API_URL}.")
        print("Is the 'python src/api/app.py' script still running in another terminal?")

if __name__ == "__main__":
    test_prediction()