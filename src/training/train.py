import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms as T
import os
import sys
from tqdm import tqdm

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.config_loader import load_config
from src.data.dataset import AuraScanDataset
from src.models.model import AuraScanModel

def collate_fn(batch):
    """Custom collate function to handle variable-sized targets."""
    return tuple(zip(*batch))

def train_one_epoch(model, dataloader, optimizer, criterion_severity, criterion_bbox, device):
    """
    Performs one full training pass over the training dataset.
    """
    model.train()  # Set the model to training mode
    total_loss = 0.0
    
    for images, targets in tqdm(dataloader, desc="Training"):
        images = list(image.to(device) for image in images)
        
        # --- Prepare targets ---
        # Note: Bbox targets need special handling for this simplified model
        # We will average the boxes for images with multiple damages for this example
        target_severities = torch.stack([t['severity'] for t in targets]).to(device)
        
        # --- Forward Pass ---
        outputs = model(torch.stack(images))
        
        # --- Calculate Loss ---
        loss_severity = criterion_severity(outputs['severity'], target_severities)
        
        # For bbox loss, we need to handle images that might not have boxes
        # This is a simplified approach. A true object detector has a more complex loss.
        bbox_targets = [t['boxes'] for t in targets]
        # For simplicity, we'll only compute bbox loss for images that have boxes and one box prediction
        # We'll take the first box if multiple are present
        valid_preds = []
        valid_targets = []
        for i, boxes in enumerate(bbox_targets):
            if len(boxes) > 0:
                valid_preds.append(outputs['boxes'][i].unsqueeze(0))
                valid_targets.append(boxes[0].unsqueeze(0).to(device)) # Just use the first box

        if valid_preds:
            loss_bbox = criterion_bbox(torch.cat(valid_preds), torch.cat(valid_targets))
        else:
            loss_bbox = torch.tensor(0.0).to(device) # No boxes in this batch to train on
        
        # Combine the losses. We can weight them if one task is more important.
        loss = loss_severity + loss_bbox
        
        # --- Backward Pass and Optimization ---
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def validate_one_epoch(model, dataloader, criterion_severity, criterion_bbox, device):
    """
    Performs one full validation pass over the validation dataset.
    """
    model.eval() # Set the model to evaluation mode
    total_loss = 0.0
    
    with torch.no_grad(): # Disable gradient calculations for validation
        for images, targets in tqdm(dataloader, desc="Validating"):
            images = list(image.to(device) for image in images)
            target_severities = torch.stack([t['severity'] for t in targets]).to(device)
            
            outputs = model(torch.stack(images))
            
            loss_severity = criterion_severity(outputs['severity'], target_severities)
            
            bbox_targets = [t['boxes'] for t in targets]
            valid_preds = []
            valid_targets = []
            for i, boxes in enumerate(bbox_targets):
                if len(boxes) > 0:
                    valid_preds.append(outputs['boxes'][i].unsqueeze(0))
                    valid_targets.append(boxes[0].unsqueeze(0).to(device))

            if valid_preds:
                loss_bbox = criterion_bbox(torch.cat(valid_preds), torch.cat(valid_targets))
            else:
                loss_bbox = torch.tensor(0.0).to(device)
            
            loss = loss_severity + loss_bbox
            total_loss += loss.item()
            
    return total_loss / len(dataloader)

def main():
    """
    Main function to run the model training pipeline.
    """
    print("--- Starting Model Training Pipeline ---")
    
    # --- 1. Load Configuration ---
    config_path = os.path.join(project_root, "configs", "config.yaml")
    config = load_config(config_path)
    
    # --- 2. Setup Device ---
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- 3. Prepare Datasets and DataLoaders ---
    data_transforms = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    annotations_csv_path = os.path.join(project_root, 'data', 'processed', 'annotations.csv')
    
    train_dataset = AuraScanDataset(annotations_csv_path, project_root, 'training', data_transforms)
    val_dataset = AuraScanDataset(annotations_csv_path, project_root, 'validation', data_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    # --- 4. Initialize Model, Optimizer, and Loss Functions ---
    num_severity_classes = len(train_dataset.class_to_idx)
    model = AuraScanModel(num_classes_severity=num_severity_classes, pretrained=True).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    # CrossEntropyLoss is standard for classification tasks
    criterion_severity = nn.CrossEntropyLoss()
    # L1Loss (Mean Absolute Error) is a good choice for bounding box regression
    criterion_bbox = nn.L1Loss()
    
    # --- 5. Training Loop ---
    best_val_loss = float('inf')
    epochs = config['training']['epochs']
    
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion_severity, criterion_bbox, device)
        val_loss = validate_one_epoch(model, val_loader, criterion_severity, criterion_bbox, device)
        
        print(f"Epoch {epoch+1} Summary: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # --- 6. Save the Best Model (Checkpointing) ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_save_path = os.path.join(project_root, config['output']['model_save_path'], 'best_model.pth')
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            print(f"Validation loss improved. Saving model to {model_save_path}")

    print("\n--- Training Complete ---")

if __name__ == '__main__':
    # This check prevents the script from running automatically if imported elsewhere
    main()