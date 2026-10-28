radar-target-classifier/
│
├── data/                  # Data directory (contains raw/processed outputs)
│   ├── raw/               # Raw generated or downloaded radar sequence arrays (.npy / .csv)
│   └── processed/         # Scaled features and train/test splits
│
├── notebooks/             # Jupyter notebooks for EDA and rapid experimentation
│   └── 01_eda_exploration.ipynb
│
├── src/                   # Core source code package (Modular Python files)
│   ├── __init__.py
│   ├── data_loader.py     # Data simulation or loading routines
│   ├── features.py        # Statistical feature extraction for SVM & RF
│   ├── models.py          # Class definitions for SVM, Random Forest, RNN, and LSTM
│   └── evaluate.py        # Metrics computation (Accuracy, Precision, Recall, Confusion Matrix)
│
├── outputs/               # Saved model artifacts, evaluation metrics, and generated slides
│   ├── models/            # Saved .pkl or .pt model weights
│   └── figures/           # Confusion matrices and performance comparison plots
││
├── tests/                 # Unit tests for data shapes and model pipeline components
│   └── test_pipeline.py
│
├── .gitignore
├── requirements.txt       # Dependencies (numpy, pandas, scikit-learn, torch, matplotlib, python-pptx)
├── README.md              # Professional documentation detailing architecture and results
└── main.py                # Main orchestration script to run the entire pipeline end-to-end


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