import os
import numpy as np
import pandas as pd

def load_astra_dataset(csv_path="data/raw/astra_dataset.csv"):
    """
    Loads and parses the Micro-Doppler Aerial Classification Dataset.
    Expected structure: flattened CSV where each row corresponds to a sample 
    containing time-series data across 100 timesteps with 3 features each.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}. Please verify path.")
    
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"Dataset loaded successfully. Shape: {df.shape}")
    return df