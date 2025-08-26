import torch
import torchvision.transforms as T
from PIL import Image
import os
import sys
import numpy as np

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.models.model import AuraScanModel

def _estimate_cost(severity: str) -> dict:
    """
    A simple business rule engine to estimate repair cost based on severity.
    """
    if severity == 'minor':
        return {'min': 50, 'max': 250}
    elif severity == 'moderate':
        return {'min': 250, 'max': 800}
    elif severity == 'severe':
        return {'min': 800, 'max': 2500}
    else: # For 'unspecified' or any other case
        return {'min': 100, 'max': 1000}

class Predictor:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            print("--- Initializing Predictor Singleton ---")
            cls._instance = super(Predictor, cls).__new__(cls)
            
            cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Predictor using device: {cls.device}")

            cls.model = AuraScanModel(num_classes_severity=4, pretrained=False)
            
            model_path = os.path.join(project_root, 'models', 'best_model.pth')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
                
            cls.model.load_state_dict(torch.load(model_path, map_location=cls.device))
            cls.model.to(cls.device)
            cls.model.eval()
            print("Model loaded and set to evaluation mode.")

            cls.transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            cls.idx_to_class = {0: 'minor', 1: 'moderate', 2: 'severe', 3: 'unspecified'}

        return cls._instance

    def predict(self, image_file_stream):
        """
        Takes an image file stream and returns a rich analysis with post-processing logic.
        """
        try:
            image = Image.open(image_file_stream).convert("RGB")
            
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(image_tensor)
                
            # --- Get Raw Predictions ---
            severity_logits = outputs['severity']
            severity_probs = torch.nn.functional.softmax(severity_logits, dim=1)
            top_prob, top_idx = torch.max(severity_probs, 1)
            
            predicted_class = self.idx_to_class[top_idx.item()]
            confidence = top_prob.item()
            predicted_bbox = outputs['boxes'][0].cpu().tolist()
            
            # --- STRATEGIC FIX: Post-processing logic ---
            # If a box is found, the severity cannot be 'unspecified'.
            # We will determine a more logical severity based on the box size.
            if predicted_class == 'unspecified' and predicted_bbox:
                # The box coordinates are in a 224x224 space.
                x_min, y_min, x_max, y_max = predicted_bbox
                box_area = (x_max - x_min) * (y_max - y_min)
                
                # Rule: If the box covers more than 20% of the image area, classify it as 'severe'.
                if box_area > (224 * 224 * 0.20): 
                    predicted_class = 'severe'
                # Rule: If the box covers more than 5% of the image, classify it as 'moderate'.
                elif box_area > (224 * 224 * 0.05):
                    predicted_class = 'moderate'
                # Otherwise, it's a small area.
                else:
                    predicted_class = 'minor'
            
            # --- Get updated cost estimate from our business rule engine ---
            cost_range = _estimate_cost(predicted_class)

            # --- Format the final, smarter result ---
            result = {
                'success': True,
                'totalDamages': 1, # Acknowledge we find one primary area
                'overallSeverity': predicted_class,
                'confidence': f"{confidence:.2f}",
                'costRange': cost_range,
                'damages': [
                    {
                        'id': 'dmg-1',
                        'type': 'Primary Damage Area',
                        'location': 'Detected by AI',
                        'severity': predicted_class,
                        'estimatedCost': cost_range,
                        'coordinates': [predicted_bbox]
                    }
                ]
            }
            return result
            
        except Exception as e:
            print(f"ERROR during prediction: {e}")
            return {'success': False, 'error': str(e)}

if __name__ == '__main__':
    # A simple test to verify the predictor works
    predictor = Predictor()
    
    test_image_path = "test_image.jpg"
    if os.path.exists(test_image_path):
        with open(test_image_path, "rb") as f:
            prediction = predictor.predict(f)
        
        import json
        print("\n--- Predictor Test ---")
        print(json.dumps(prediction, indent=2))
    else:
        print(f"\n--- Predictor Test Skipped ---")
        print(f"Create a '{test_image_path}' in the root folder to run a test.")