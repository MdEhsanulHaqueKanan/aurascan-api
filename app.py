# This is the entry point for the Hugging Face Space.
# It simply imports the Flask app instance from our structured source code.
from src.api.app import app

# The 'app' variable is what the Hugging Face/Gunicorn server will look for.