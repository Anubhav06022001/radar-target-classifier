import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import wandb

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import extract_statistical_features
from src.models import get_svm_model, get_random_forest_model

def evaluate_traditional_model(model_type="random_forest", use_wandb=False):
    """
    Trains, evaluates, and logs either SVM or Random Forest using engineered tabular features.
    """
    if use_wandb:
        wandb.init(project="radar-target-classifier", name=f"train_{model_type}")

    # 1. Load Data
    df = pd.read_csv("data/raw/astra_dataset.csv")
    feature_cols = [str(i) for i in range(300)]
    
    # 2. Extract Tabular Features (15 summary features)
    X_tab = extract_statistical_features(df, feature_cols)
    y = df['label'].values
    
    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_tab, y, test_size=0.2, random_state=42)
    
    # 4. Scale Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Select Model
    if model_type == "random_forest":
        model = get_random_forest_model(n_estimators=100)
    elif model_type == "svm":
        model = get_svm_model(C=1.0, kernel='rbf')
    else:
        raise ValueError("Unknown model type. Choose 'random_forest' or 'svm'.")
        
    print(f"\nTraining {model_type.upper()}...")
    model.fit(X_train_scaled, y_train)
    
    # 6. Evaluate
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n--- {model_type.upper()} RESULTS ---")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Bird", "Drone", "Aircraft", "Stealth UAV"]))
    
    if use_wandb:
        wandb.log({"accuracy": acc})
        wandb.finish()
        
    return model, acc

if __name__ == "__main__":
    # Run Random Forest evaluation first
    # evaluate_traditional_model(model_type="random_forest", use_wandb=False)
    evaluate_traditional_model(model_type="svm", use_wandb=False)