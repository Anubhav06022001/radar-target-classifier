# --- RANDOM FOREST FEATURE IMPORTANCE VISUALIZATION ---
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.features import extract_statistical_features

os.makedirs("outputs/figures", exist_ok=True)

# 1. Load data and extract tabular features
df = pd.read_csv("data/raw/astra_dataset.csv")
feature_cols = [str(i) for i in range(300)]
X_tab = extract_statistical_features(df, feature_cols)
y = df['label'].values

# 2. Train/Test split & scaling
X_train, X_test, y_train, y_test = train_test_split(X_tab, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 3. Fit Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)

# 4. Map feature names (3 physical channels * 5 statistics)
channels = ["Amplitude", "Velocity", "Energy"]
stats = ["Mean", "Std", "Max", "Min", "RMS"]
feature_names = [f"{ch} - {st}" for ch in channels for st in stats]

# 5. Plot Importances
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices], palette='viridis')
plt.title("Random Forest Feature Importances for Radar Target Classification", fontsize=12, fontweight='bold')
plt.xlabel("Importance Score", fontsize=10)
plt.ylabel("Engineered Statistical Feature", fontsize=10)
plt.tight_layout()

output_path = "outputs/figures/rf_feature_importance.png"
plt.savefig(output_path, dpi=300)
plt.show()

print(f"Feature importance plot saved successfully to {output_path}")