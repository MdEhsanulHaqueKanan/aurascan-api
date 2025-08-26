import os
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- ROBUSTNESS FIX: Add project to the Python path ---
# This ensures that we can always find our modules, whether running locally or on a server.
# It navigates up from src/api to the project root 'aurascan-api-deployment'.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    # We use insert(0) to make sure our project's src is checked first
    sys.path.insert(0, project_root)

# Now that the path is set, we can use a direct, absolute import from the src root.
# This works both locally and in production.
from src.api.predictor import Predictor

# --- Initialize the Flask App ---
app = Flask(__name__)
# Enable CORS for all routes, allowing our frontend to communicate with this API
CORS(app)

# --- Initialize our Predictor Singleton ---
# This will load the model into memory when the app starts
predictor = Predictor()
print("\n--- Flask App Initialized ---")

@app.route("/ping", methods=["GET"])
def ping():
    """A simple endpoint to check if the server is running."""
    return jsonify({"status": "ok", "message": "Server is alive!"})

@app.route("/analyze", methods=["POST"])
def analyze():
    """
    The main endpoint for analyzing an uploaded car image.
    """
    # Check if an image was uploaded
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file part in the request"}), 400
    
    file = request.files['file']
    
    # Check if the file is empty
    if file.filename == '':
        return jsonify({"success": False, "error": "No file selected for uploading"}), 400
        
    if file:
        try:
            # Pass the file stream directly to the predictor
            prediction = predictor.predict(file.stream)
            
            return jsonify(prediction)

        except Exception as e:
            return jsonify({"success": False, "error": f"An error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    # This block is for local testing only.
    # For production, a proper WSGI server like Gunicorn will be used.
    app.run(host="0.0.0.0", port=5000, debug=True)