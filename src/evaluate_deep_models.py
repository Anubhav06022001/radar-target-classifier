import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.models import RadarRNN, RadarLSTM

def train_and_evaluate_sequence_model(model_type="lstm", epochs=15, batch_size=32):

    print(f"--- Training Deep Sequence Model: {model_type.upper()} ---")
    
    # 1. Load Data
    df = pd.read_csv("data/raw/astra_dataset.csv")
    feature_cols = [str(i) for i in range(300)]
    X_flat = df[feature_cols].values
    y = df['label'].values
    
    # 2. Reshape to 3D Tensor: (Samples, 100 timesteps, 3 features)
    X_3d = X_flat.reshape(-1, 100, 3)
    
    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_3d, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. Standardize features across time steps per channel
    # Reshape to 2D for scaling, then back to 3D
    nsamples, nx, ny = X_train.shape
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, ny)).reshape(nsamples, nx, ny)
    
    nsamples_test, nx_test, ny_test = X_test.shape
    X_test_scaled = scaler.transform(X_test.reshape(-1, ny_test)).reshape(nsamples_test, nx_test, ny_test)
    
    # 5. Convert to PyTorch Tensors & DataLoaders
    train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 6. Initialize Model, Loss, and Optimizer
    if model_type == "rnn":
        model = RadarRNN(input_dim=3, hidden_dim=64, num_classes=4)
    elif model_type == "lstm":
        model = RadarLSTM(input_dim=3, hidden_dim=64, num_classes=4)
    else:
        raise ValueError("Choose 'rnn' or 'lstm'.")
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 7. Training Loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")
        
    # 8. Evaluation Loop
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
    acc = correct / total
    print(f"\n--- {model_type.upper()} RESULTS ---")
    print(f"Test Accuracy: {acc:.4f}\n")
    return model, acc

if __name__ == "__main__":
    # train_and_evaluate_sequence_model(model_type="lstm", epochs=10, batch_size=32)
    train_and_evaluate_sequence_model(model_type="rnn", epochs=10, batch_size=32)