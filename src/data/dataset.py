import torch
import pandas as pd
from PIL import Image
import ast # Used for safely evaluating string representations of lists

class AuraScanDataset(torch.utils.data.Dataset):
    """
    Custom PyTorch Dataset for the AuraScanAI project.
    Reads the processed annotations.csv file and prepares data for the model.
    """
    def __init__(self, annotations_file, project_root, split, transform=None):
        """
        Args:
            annotations_file (str): Path to the csv file with annotations.
            project_root (str): The absolute path to the project's root directory.
            split (str): The dataset split to use, 'training' or 'validation'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.project_root = project_root
        self.transform = transform
        
        # Load the annotations and filter for the specified split
        full_df = pd.read_csv(annotations_file)
        self.df = full_df[full_df['split'] == split].reset_index(drop=True)
        
        # Create a mapping from class names to integer indices
        self.class_to_idx = {'minor': 0, 'moderate': 1, 'severe': 2, 'unspecified': 3}
        
        print(f"Initialized dataset for split: '{split}'. Found {len(self.df)} records.")

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.df)

    def __getitem__(self, idx):
        """
        Fetches the sample at the given index.

        Args:
            idx (int): The index of the sample.

        Returns:
            tuple: (image, target) where target is a dictionary containing
                   bounding boxes, severity labels, and box labels.
        """
        # --- FIX: Import torch here to ensure it's available in the worker process ---
        import torch

        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Get the record for the given index
        record = self.df.iloc[idx]
        
        # Load the image
        # Note: The image_path in the CSV is absolute, so we don't need to join it
        image_path = record['image_path']
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"FATAL ERROR: Image not found at {image_path}. Check your annotations.csv file.")
            # Return a dummy sample to avoid crashing the whole training loop
            # A more robust solution might involve cleaning the CSV beforehand
            return torch.randn(3, 224, 224), {}

        # Parse the bounding boxes from their string representation
        # ast.literal_eval is a safe way to do this
        bboxes_str = record['damage_locations']
        bboxes = ast.literal_eval(bboxes_str)
        
        # Convert severity label string to an integer index
        severity_label_str = record['severity_label']
        severity_idx = self.class_to_idx[severity_label_str]
        
        # --- Prepare the target dictionary ---
        target = {}
        # Bounding boxes must be a FloatTensor
        target['boxes'] = torch.tensor(bboxes, dtype=torch.float32)
        # For this project, all boxes are simply "damage", so we use label 1 for all.
        # Label 0 is typically reserved for the background.
        target['labels'] = torch.ones((len(bboxes),), dtype=torch.int64)
        # Severity label must be a LongTensor for classification
        target['severity'] = torch.tensor(severity_idx, dtype=torch.long)

        # Apply transformations if they exist
        if self.transform:
            image = self.transform(image)
            
        return image, target