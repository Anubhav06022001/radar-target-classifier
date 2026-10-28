# Radar Target Classification using Micro-Doppler Signatures

A production-grade machine learning and deep learning pipeline for classifying aerial targets (Birds, Drones, Aircraft, Stealth UAVs) using radar micro-Doppler time-series data. Developed for robotics navigation and defense applications.

---

## 📂 Repository Architecture
```text
radar-target-classifier/
├── data/
│   └── raw/
│       └── astra_dataset.csv       # 2800 samples, 4 classes, 300 features
├── src/
│   ├── data_loader.py              # Data ingestion & integrity checks
│   ├── features.py                 # Statistical feature extraction (Mean, Std, Max, Min, RMS)
│   ├── models.py                   # Model architectures (SVM, Random Forest, RNN, LSTM)
│   └── evaluate_deep_models.py     # PyTorch training & evaluation loops for sequence models
├── outputs/
│   └── figures/                    # Publication-grade EDA & performance plots
├── README.md
└── requirements.txt
```

---

## 🚀 Getting Started Locally

Follow these steps to set up your environment and run the pipeline locally.

### 1. Clone the Repository & Set Up Virtual Environment
```bash
git clone [https://github.com/your-username/radar-target-classifier.git](https://github.com/your-username/radar-target-classifier.git)
cd radar-target-classifier
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run Traditional ML Models (SVM & Random Forest)
```bash
python src/evaluate.py
```

### 4. Run Deep Learning Models (RNN & LSTM)
```bash
python src/evaluate_deep_models.py
```

---

## 📊 Benchmark Results

We evaluated traditional machine learning models (using engineered statistical features) against deep recurrent sequence models (ingesting raw 3D tensors):

| Model Architecture | Input Data Format | Test Accuracy |
| :--- | :--- | :---: |
| **Random Forest** | Tabular (15 Statistical Features) | **100.00%** |
| **Support Vector Machine (SVM)** | Tabular (15 Statistical Features) | **100.00%** |
| **LSTM (PyTorch)** | Raw 3D Tensors (`100 × 3`) | **99.29%** |
| **Vanilla RNN (PyTorch)** | Raw 3D Tensors (`100 × 3`) | **98.04%** |