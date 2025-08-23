import torch
import torch.nn as nn
import timm

class AuraScanModel(nn.Module):
    """
    A multi-task Vision Transformer model for AuraScanAI.

    This model uses a pre-trained Vision Transformer (ViT) as a backbone
    and adds two separate 'heads' on top for two distinct tasks:
    1.  An object detection head to predict bounding boxes for damages.
    2.  A classification head to predict the overall severity of the damage.
    """
    def __init__(self, num_classes_severity, pretrained=True):
        """
        Initializes the model.

        Args:
            num_classes_severity (int): The number of severity classes 
                                       (e.g., 4 for minor, moderate, severe, unspecified).
            pretrained (bool): Whether to use a pre-trained ViT backbone from timm.
        """
        super(AuraScanModel, self).__init__()
        
        # --- 1. Load the Pre-trained ViT Backbone ---
        # We use a standard, powerful ViT model. 'vit_base_patch16_224' is a great choice.
        # pretrained=True downloads weights trained on ImageNet.
        # We set num_classes=0 because we don't want the original classification head.
        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=0)
        
        # Get the number of output features from the backbone
        # For this ViT model, it's typically 768.
        n_features = self.backbone.num_features
        
        # --- 2. Define the Severity Classification Head ---
        # A simple linear layer is sufficient. It takes the ViT's output 
        # and maps it to the number of severity classes.
        self.severity_head = nn.Linear(n_features, num_classes_severity)
        
        # --- 3. Define the Bounding Box Regression Head ---
        # This is a simplified approach for demonstration. It predicts a single bounding box.
        # A full object detector would use a more complex head (like in Faster R-CNN or YOLO).
        # This head will learn to predict the "most prominent" damage box.
        self.bbox_head = nn.Linear(n_features, 4) # Predicts the 4 coordinates (x_min, y_min, x_max, y_max)
        
        print(f"AuraScanModel initialized with a ViT backbone.")
        print(f" - Backbone output features: {n_features}")
        print(f" - Severity classification head for {num_classes_severity} classes.")
        print(f" - Bounding box regression head for a single box (4 coordinates).")

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): A batch of input images of shape (N, C, H, W).

        Returns:
            dict: A dictionary containing the model's outputs for each head.
        """
        # --- Pass input through the backbone ---
        # The output 'features' is a tensor of shape (batch_size, num_features)
        # It represents the high-level understanding of each image in the batch.
        features = self.backbone(x)
        
        # --- Pass the features through each head independently ---
        severity_logits = self.severity_head(features)
        bbox_predictions = self.bbox_head(features)
        
        # --- Return the outputs in a structured dictionary ---
        # This makes it easy to calculate the loss for each task separately.
        outputs = {
            'severity': severity_logits,
            'boxes': bbox_predictions
        }
        
        return outputs

if __name__ == '__main__':
    # A simple test to verify the model works as expected
    
    # Create a dummy batch of 4 images (3 channels, 224x224 pixels)
    dummy_images = torch.randn(4, 3, 224, 224)
    
    # Instantiate the model (4 severity classes, don't download weights for this test)
    model = AuraScanModel(num_classes_severity=4, pretrained=False)
    
    # Perform a forward pass
    outputs = model(dummy_images)
    
    # Print the shapes of the outputs to verify they are correct
    print("\n--- Model Test ---")
    print(f"Input shape: {dummy_images.shape}")
    print("\nOutput shapes:")
    for name, tensor in outputs.items():
        print(f" - {name}: {tensor.shape}")
        
    # Expected Output Check:
    # - severity: (4, 4) -> Batch of 4 images, 4 class scores per image
    # - boxes: (4, 4) -> Batch of 4 images, 4 coordinates per image for one box
    
    print("\nModel structure test complete.")