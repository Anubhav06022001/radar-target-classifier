import numpy as np
import pandas as pd

def extract_statistical_features(df, feature_cols):
    """
    Transforms 300 flat time-series columns (100 timesteps x 3 features) 
    into a rich tabular feature set for SVM and Random Forest.
    
    For each of the 3 physical channels (Amplitude, Velocity, Energy), 
    we extract 5 statistical descriptors: Mean, Std, Max, Min, and RMS.
    Total features per sample = 3 channels * 5 descriptors = 15 summary features.
    """
    print("Extracting statistical features for SVM & Random Forest...")
    
    # 1. Extract raw feature matrix and reshape to (Samples, 100 timesteps, 3 channels)
    X_flat = df[feature_cols].values      # .values converts pandas DataFrame to NumPy array
    X_3d = X_flat.reshape(-1, 100, 3)
    
    tabular_features = []
    
    for sample in X_3d:
        sample_feats = []
        # Loop through each of the 3 channels (Amplitude, Velocity, Energy)
        for channel_idx in range(3):
            channel_data = sample[:, channel_idx]
            
            mean_val = np.mean(channel_data)
            std_val = np.std(channel_data)
            max_val = np.max(channel_data)
            min_val = np.min(channel_data)
            rms_val = np.sqrt(np.mean(channel_data**2)) # Root Mean Square
            
            sample_feats.extend([mean_val, std_val, max_val, min_val, rms_val])
            
        tabular_features.append(sample_feats)
        
    X_tabular = np.array(tabular_features)
    print(f"Tabular feature extraction complete. New shape: {X_tabular.shape}")
    
    return X_tabular