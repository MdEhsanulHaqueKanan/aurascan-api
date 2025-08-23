import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys

# Add project root to the Python path to allow importing from src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.config_loader import load_config

def process_severity_dataset(base_path: str, config: dict) -> list:
    """
    Processes the folder-based severity dataset.

    Args:
        base_path (str): The absolute path to the project root.
        config (dict): The project configuration dictionary.

    Returns:
        list: A list of dictionaries, where each dictionary represents an image record.
    """
    print("Processing Severity Dataset...")
    records = []
    severity_path = os.path.join(base_path, 'data', 'raw', config['data']['severity_dataset'], 'data3a')
    
    for split in ['training', 'validation']:
        split_path = os.path.join(severity_path, split)
        if not os.path.exists(split_path):
            continue
            
        for severity in os.listdir(split_path):
            severity_path_full = os.path.join(split_path, severity)
            if not os.path.isdir(severity_path_full):
                continue
                
            # Extract the label, e.g., '01-minor' -> 'minor'
            label = severity.split('-')[1]
            
            for image_name in tqdm(os.listdir(severity_path_full), desc=f"Processing {split}/{severity}"):
                image_path = os.path.join(severity_path_full, image_name)
                records.append({
                    'image_path': image_path,
                    'damage_locations': [], # This dataset has no location data
                    'severity_label': label,
                    'split': split,
                    'source': 'severity_dataset'
                })
    return records

def process_vehide_dataset(base_path: str, config: dict) -> list:
    """
    Processes the JSON-based VehiDE dataset, converting polygons to bounding boxes.

    Args:
        base_path (str): The absolute path to the project root.
        config (dict): The project configuration dictionary.

    Returns:
        list: A list of dictionaries, where each dictionary represents an image record.
    """
    print("\nProcessing VehiDE Dataset...")
    records = []
    vehide_path = os.path.join(base_path, 'data', 'raw', config['data']['vehide_dataset'])
    
    for split in ['train', 'val']:
        # Determine the correct annotation file and image folder for the split
        if split == 'train':
            json_file = '0Train_via_annos.json'
            image_folder = os.path.join(vehide_path, 'image', 'image')
        else: # split == 'val'
            json_file = '0Val_via_annos.json'
            image_folder = os.path.join(vehide_path, 'validation', 'validation')
            
        json_path = os.path.join(vehide_path, json_file)
        if not os.path.exists(json_path):
            print(f"Warning: Annotation file not found for split '{split}'. Skipping.")
            continue
        
        with open(json_path, 'r') as f:
            annotations = json.load(f)
            
        for image_key, image_info in tqdm(annotations.items(), desc=f"Processing VehiDE {split} split"):
            image_path = os.path.join(image_folder, image_info['name'])
            bboxes = []
            
            for region in image_info['regions']:
                all_x = region['all_x']
                all_y = region['all_y']
                
                # Convert polygon to a bounding box
                x_min = np.min(all_x)
                y_min = np.min(all_y)
                x_max = np.max(all_x)
                y_max = np.max(all_y)
                bboxes.append([int(x_min), int(y_min), int(x_max), int(y_max)])
            
            records.append({
                'image_path': image_path,
                'damage_locations': bboxes,
                'severity_label': 'unspecified', # This dataset has no severity data
                'split': 'training' if split == 'train' else 'validation', # Standardize split name
                'source': 'vehide_dataset'
            })
            
    return records

def main():
    """
    Main function to run the data processing pipeline.
    """
    print("--- Starting Data Processing Pipeline ---")
    
    # Load configuration
    config_path = os.path.join(project_root, "configs", "config.yaml")
    config = load_config(config_path)

    # Process both datasets
    severity_records = process_severity_dataset(project_root, config)
    vehide_records = process_vehide_dataset(project_root, config)
    
    # Combine, convert to DataFrame, and shuffle
    all_records = severity_records + vehide_records
    df = pd.DataFrame(all_records)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True) # Shuffle the dataset
    
    # Define and create the output directory
    processed_path = os.path.join(project_root, 'data', 'processed')
    os.makedirs(processed_path, exist_ok=True)
    
    # Save the final annotations file
    output_csv_path = os.path.join(processed_path, 'annotations.csv')
    df.to_csv(output_csv_path, index=False)
    
    print(f"\n--- Pipeline Complete ---")
    print(f"Total records processed: {len(df)}")
    print(f"Annotations saved to: {output_csv_path}")
    print("\nFirst 5 rows of the final DataFrame:")
    print(df.head())

if __name__ == '__main__':
    main()