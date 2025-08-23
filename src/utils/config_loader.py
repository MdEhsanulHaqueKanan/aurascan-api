import yaml
from typing import Dict

def load_config(config_path: str = "configs/config.yaml") -> Dict:
    """
    Loads a YAML configuration file.

    Args:
        config_path (str): The path to the configuration file.

    Returns:
        Dict: A dictionary containing the configuration parameters.
        
    Raises:
        FileNotFoundError: If the config file is not found.
        yaml.YAMLError: If the config file is malformed.
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        print("Configuration file loaded successfully from src/utils/config_loader.py")
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        raise
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        raise