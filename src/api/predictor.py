import torch
import torchvision.transforms as T
from PIL import Image
import os
import sys

# Add project root to the Python path to allow importing from src.models
# This assumes the script is run from the project root.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.models.model import AuraScanModel

class Predictor:
    """
    Singleton class to load the AuraScanModel and make predictions.
    This ensures the model is loaded into memory only once.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            print("--- Initializing Predictor Singleton ---")
            cls._instance = super(Predictor, cls).__new__(cls)
            
            # --- Load Model ---
            cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Predictor using device: {cls.device}")

            # The number of classes our saved model was trained on
            # 4 classes: minor, moderate, severe, unspecified
            cls.model = AuraScanModel(num_classes_severity=4, pretrained=False)
            
            # Load the saved weights
            model_path = os.path.join(project_root, 'models', 'best_model.pth')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
                
            cls.model.load_state_dict(torch.load(model_path, map_location=cls.device))
            cls.model.to(cls.device)
            cls.model.eval() # Set model to evaluation mode
            print("Model loaded and set to evaluation mode.")

            # --- Define Image Transformations ---
            # This must be the SAME as the transformations used during training
            cls.transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # --- Define Class Mapping ---
            # This is the reverse of the mapping in our Dataset class
            cls.idx_to_class = {0: 'minor', 1: 'moderate', 2: 'severe', 3: 'unspecified'}

        return cls._instance

    def predict(self, image_file_stream):
        """
        Takes an image file stream, preprocesses it, and returns the model's prediction.

        Args:
            image_file_stream: The image file stream object from Flask.

        Returns:
            dict: A dictionary containing the prediction results.
        """
        try:
            # --- FIX: Load image directly from the file stream ---
            image = Image.open(image_file_stream).convert("RGB")
            
            # Preprocess the image
            image_tensor = self.transform(image).unsqueeze(0) # Add batch dimension
            image_tensor = image_tensor.to(self.device)
            
            # Get model prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                
            # --- Process Severity Prediction ---
            severity_logits = outputs['severity']
            # Apply softmax to get probabilities
            severity_probs = torch.nn.functional.softmax(severity_logits, dim=1)
            # Get the top class index and its probability
            top_prob, top_idx = torch.max(severity_probs, 1)
            
            predicted_class = self.idx_to_class[top_idx.item()]
            confidence = top_prob.item()
            
            # --- Process Bbox Prediction ---
            # Note: Bbox values are predicted relative to the image size.
            # For this simplified model, it predicts one box.
            predicted_bbox = outputs['boxes'][0].cpu().tolist()

            # --- Format the Final Result ---
            result = {
                'success': True,
                'predictions': {
                    'severity': {
                        'class': predicted_class,
                        'confidence': f"{confidence:.4f}"
                    },
                    'bounding_box': {
                        'box_coordinates': predicted_bbox
                        # We could add logic to scale these back to original image size
                    }
                }
            }
            return result
            
        except Exception as e:
            print(f"ERROR during prediction: {e}")
            return {'success': False, 'error': str(e)}

if __name__ == '__main__':
    # A simple test to verify the predictor works
    # You would need a sample image named 'test_image.jpg' in the root directory
    predictor = Predictor()
    
    test_image_path = "test_image.jpg" # Create a dummy image for this
    if os.path.exists(test_image_path):
        with open(test_image_path, "rb") as f:
            # For local test, we still pass the stream
            prediction = predictor.predict(f)
        
        print("\n--- Predictor Test ---")
        import json
        print(json.dumps(prediction, indent=2))
    else:
        print(f"\n--- Predictor Test Skipped ---")
        print(f"Create a '{test_image_path}' in the root folder to run a test.")