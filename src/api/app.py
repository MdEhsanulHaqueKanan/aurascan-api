from flask import Flask, request, jsonify
from flask_cors import CORS
from .predictor import Predictor

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
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file part in the request"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"success": False, "error": "No file selected for uploading"}), 400
        
    if file:
        try:
            prediction = predictor.predict(file.stream)
            
            return jsonify(prediction)

        except Exception as e:
            return jsonify({"success": False, "error": f"An error occurred: {str(e)}"}), 500
