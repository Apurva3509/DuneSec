# DDoS Attack Detection System

![Project Banner](src/ddos.png)

## Introduction
In this project, we develop a **machine learning solution** for real-time DDoS attack detection using **Random Forest classification**. The dataset consists of network traffic flows collected using **CICFlowMeterV3**, containing both benign traffic and traffic from **DDoS attacks**. The goal is to **accurately distinguish** between these two classes and deploy an efficient detection system.

---
# Table of Contents

1. [Quick Start Guide](#quick-start-guide-aka-how-to-run-the-code)
2. [Assignment Breakdown](#assignment-breakdown)
3. [Implementation Details](#implementation-details)
4. [Using the API](#using-the-api)
5. [Results & Evaluation](#results--evaluation)
6. [Monitoring & Logging](#monitoring--logging)
7. [Documentation](#documentation)
8. [Future Improvements](#future-improvements)

---



## Quick Start Guide (aka How to run the code):

#### 1. **Setup Environment**
```bash
# Install Git LFS - if not already installed
brew install git-lfs
git lfs install

# Clone the repository
git clone https://github.com/Apurva3509/DuneSec.git
cd DuneSec-main

# Create virtual environment - OPTIONAL
python -m venv venv
source venv/bin/activate

# Install dependencies - OPTIONAL
pip install -r requirements.txt
```

#### 2. **Directory Structure**
```bash
# Create necessary directories
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/test
mkdir -p models/preprocessors
mkdir -p reports/figures
mkdir -p logs
```
> **Note:** Place your network traffic CSV file in `data/raw/network_traffic.csv`.

#### 3. **Run the Pipeline**
```bash
# Step 1: Split the data into train (80%) and test (20%) sets
python main.py --mode split

# Step 2: Train the model
python main.py --mode train

# Step 3: Evaluate the model on test set
python main.py --mode test

```
---

## Assignment Breakdown

### Data Exploration & Analysis
Before building the model, an **Exploratory Data Analysis (EDA)** was performed to gain insights into the dataset:
- **Descriptive Statistics**: Understanding the distribution of features.
- **Data Quality Issues**:
  - Presence of missing values and outliers.
  - Identification of highly correlated features for removal.
- **Visualizations**:
  - Class distribution (`DDoS: 56.7%, BENIGN: 43.3%`).
  - Feature importance and correlation heatmaps.

For details, refer to: 
1. [`Preliminary model notebook`](notebooks/DDoS-detection.ipynb).
2. [`EDA Notebook`](notebooks/data_eda-v2.ipynb).
---

### Modeling Strategy
The chosen model for detecting DDoS attacks is **Random Forest**, due to:
- **High interpretability**: Feature importance tracking.
- **Robustness to noisy data**.
- **Fast inference speed** (~5.3ms per prediction).

## Implementation Details

### 1. Data Pipeline
- **Data Ingestion**: 
  - Implemented network flow data collection from csv file
  - Built robust data loading with error handling and validation
  - Automated data quality checks for missing values and anomalies

- **Preprocessing Pipeline**:
  - Feature scaling using StandardScaler
  - Label encoding for target variable
  - Feature selection based on correlation analysis
  - Data split: 80% training, 20% testing (configured in config.yaml)

### 2. Model Development
- **Algorithm Selection**:
  - Evaluated multiple models (Random Forest, XGBoost, Neural Networks, Logistic regression)
  - Selected Random Forest for best balance of accuracy and inference speed
  - Implemented 5-fold cross-validation for robust evaluation

- **Hyperparameter Optimization**:
  - Grid search for parameter tuning
  - Optimized for F1-score
  - Parameters tracked using MLflow

### 3. Evaluation System
- **Performance Metrics**:
  - Monitoring of accuracy, precision, recall
  - Confusion matrix analysis
  - ROC curve and AUC calculation

- **Visualization Pipeline**:
  - Automated generation of performance plots
  - Feature importance visualization
  - Prediction distribution analysis

### 4. Production Pipeline
- **Model Serving**:
  - Joblib serialization for model artifacts
  - Preprocessor versioning
  - Batch prediction capability

### 5. API Implementation
- **FastAPI Service**:
  - Real-time prediction endpoint
  - Model artifact management
  - Input validation
  - Error handling
  - Performance monitoring

- **Deployment Options**:
  - Local development server
  - Docker containerization

- **Testing Tools**:
  - Interactive test script
  - Swagger UI documentation
  - Python client implementation

### Using the API

1. **Start the API Server**:
```bash
# Ensure model is trained first
python main.py --mode train

# Start the API
uvicorn src.api.app:app --reload --port 8000
```

2. **Make Predictions (Open a new terminal while keeping the API running)**:
```bash
# Using test script (recommended for testing)
python src/api/test_prediction.py

# Or using curl
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": {...}}'
```

3. **View Documentation**:
- Swagger UI: http://localhost:8000/docs
- Detailed API docs: [API Documentation](docs/api.md)
- Project report slide deck: [Project Report](docs/Presentation.pdf)



#### Limitations:
- Computationally expensive for large datasets.
- May require **hyperparameter tuning** to avoid overfitting.

---


## **View Results**
- **Model artifacts**: `models/`
- **Performance metrics**: `reports/results/`
- **Visualizations**: `reports/figures/`
- **Logs**: `logs/app.log`

---

## Results & Evaluation

The model's performance was evaluated using key classification metrics:

| Metric       | Score  |
|-------------|--------|
| **Accuracy**  | 99.2%  |
| **Precision** | 98.7%  |
| **Recall**    | 99.5%  |
| **F1-Score**  | 99.1%  |
| **ROC-AUC**   | 0.998  |
| **Avg Prediction Time** | 5.3ms |

#### **Performance Evaluation**
- **Confusion Matrix**
- **ROC-AUC Curve**
- **Feature Importance Analysis**

> **All results and analysis are saved in:** `reports/results/`

---

## Monitoring & Logging
To ensure reliable performance in real-world scenarios, the system includes:
- **Real-time Performance Monitoring** (latency, accuracy)
- **Feature Importance Tracking**
- **Data Drift Detection**
- **Logging** (`logs/app.log`)

---

## Documentation

- 🔗 [**Analysis & Insights**](docs/NOTES.md) - **EDA findings and design decisions**
<!-- - 📡 [**API Documentation**](docs/api.md) - **Endpoints & usage** -->
- 📍 [**Model Architecture**](docs/model.md) - **Model structure & training pipeline**

---

## Future Improvements
- Implement **XGBoost for better performance**.
- Develop a **real-time API using FastAPI or Flask**.
- Improve **feature engineering using domain knowledge**.

---

_Developed by [Apurva Patel](https://www.patelapurva.com)_

<!-- ## Project Structure
```
.
├── config/
│   └── config.yaml
├── data/
│   ├── processed/
│   │   ├── test.csv
│   │   └── train.csv
│   └── raw/
│       ├── data-original.csv
│       └── network_traffic.csv
├── docs/
│   ├── NOTES.md
│   ├── model.md
│   └── Presentation.pdf
├── logs/
│   └── app.log
├── mlruns/
│   ├── 0/
│   │   ├── 2e9eff9f94e34abb994c9520cdae5fc0/
│   │   │   ├── artifacts/
│   │   │   ├── metrics/
│   │   │   │   ├── test_accuracy
│   │   │   │   └── test_roc_auc
│   │   │   ├── params/
│   │   │   ├── tags/
│   │   │   │   ├── mlflow.source.git.commit
│   │   │   │   ├── mlflow.source.name
│   │   │   │   ├── mlflow.source.type
│   │   │   │   └── mlflow.user
│   │   │   └── meta.yaml
│   │   ├── 3625057b68ce49c9a27ca9b56ffe27ad/
│   │   │   ├── artifacts/
│   │   │   ├── metrics/
│   │   │   │   ├── test_accuracy
│   │   │   │   └── test_roc_auc
│   │   │   ├── params/
│   │   │   ├── tags/
│   │   │   │   ├── mlflow.source.git.commit
│   │   │   │   ├── mlflow.source.name
│   │   │   │   ├── mlflow.source.type
│   │   │   │   └── mlflow.user
│   │   │   └── meta.yaml
│   │   ├── 6077e262c3d74f9e8ef2d9c405cbaf95/
│   │   │   ├── artifacts/
│   │   │   ├── metrics/
│   │   │   │   ├── test_accuracy
│   │   │   │   └── test_roc_auc
│   │   │   ├── params/
│   │   │   ├── tags/
│   │   │   │   ├── mlflow.source.git.commit
│   │   │   │   ├── mlflow.source.name
│   │   │   │   ├── mlflow.source.type
│   │   │   │   └── mlflow.user
│   │   │   └── meta.yaml
│   │   └── meta.yaml
│   └── 1/
│       ├── 081055f970e441f993a39423ed6cf9c3/
│       │   ├── artifacts/
│       │   ├── metrics/
│       │   │   ├── cv_score_mean
│       │   │   └── cv_score_std
│       │   ├── params/
│       │   │   ├── eval_metric
│       │   │   ├── learning_rate
│       │   │   ├── max_depth
│       │   │   ├── n_estimators
│       │   │   └── objective
│       │   ├── tags/
│       │   │   ├── mlflow.source.git.commit
│       │   │   ├── mlflow.source.name
│       │   │   ├── mlflow.source.type
│       │   │   └── mlflow.user
│       │   └── meta.yaml
│       ├── 27e907663df046ed8a200124db985bc6/
│       │   ├── artifacts/
│       │   ├── metrics/
│       │   │   ├── test_accuracy
│       │   │   └── test_roc_auc
│       │   ├── params/
│       │   ├── tags/
│       │   │   ├── mlflow.source.git.commit
│       │   │   ├── mlflow.source.name
│       │   │   ├── mlflow.source.type
│       │   │   └── mlflow.user
│       │   └── meta.yaml
│       ├── 511934cf31b04653a01169ddbd3c5bf1/
│       │   ├── artifacts/
│       │   ├── metrics/
│       │   │   ├── cv_score_mean
│       │   │   └── cv_score_std
│       │   ├── params/
│       │   │   ├── eval_metric
│       │   │   ├── learning_rate
│       │   │   ├── max_depth
│       │   │   ├── n_estimators
│       │   │   └── objective
│       │   ├── tags/
│       │   │   ├── mlflow.source.git.commit
│       │   │   ├── mlflow.source.name
│       │   │   ├── mlflow.source.type
│       │   │   └── mlflow.user
│       │   └── meta.yaml
│       ├── a5b66f52f27446a1b518ab7908032e80/
│       │   ├── artifacts/
│       │   ├── metrics/
│       │   ├── params/
│       │   ├── tags/
│       │   │   ├── mlflow.source.git.commit
│       │   │   ├── mlflow.source.name
│       │   │   ├── mlflow.source.type
│       │   │   └── mlflow.user
│       │   └── meta.yaml
│       ├── f55208b492c34e9cb3bf89795f6d14af/
│       │   ├── artifacts/
│       │   ├── metrics/
│       │   │   ├── cv_score_mean
│       │   │   └── cv_score_std
│       │   ├── params/
│       │   │   ├── eval_metric
│       │   │   ├── learning_rate
│       │   │   ├── max_depth
│       │   │   ├── n_estimators
│       │   │   └── objective
│       │   ├── tags/
│       │   │   ├── mlflow.source.git.commit
│       │   │   ├── mlflow.source.name
│       │   │   ├── mlflow.source.type
│       │   │   └── mlflow.user
│       │   └── meta.yaml
│       ├── fa8e4d6257cc419f911eaa4caa61d2a2/
│       │   ├── artifacts/
│       │   ├── metrics/
│       │   ├── params/
│       │   ├── tags/
│       │   │   ├── mlflow.source.git.commit
│       │   │   ├── mlflow.source.name
│       │   │   ├── mlflow.source.type
│       │   │   └── mlflow.user
│       │   └── meta.yaml
│       └── meta.yaml
├── model_artifacts/
│   ├── label_encoder.joblib
│   ├── random_forest_model.joblib
│   └── scaler.joblib
├── model_artifacts_20250224_1356/
│   ├── label_encoder.joblib
│   ├── model.joblib
│   └── scaler.joblib
├── models/
│   └── random_forest_model.joblib
├── notebooks/
│   ├── DDoS-detection.ipynb
│   ├── data_eda-v1.ipynb
│   ├── data_eda-v2.ipynb
│   ├── final-v1.ipynb
│   ├── trial-v1.ipynb
│   └── wandb-run-v1.ipynb
├── reports/
│   ├── figures/
│   │   ├── confusion_matrix_test_20250224_200818.png
│   │   ├── confusion_matrix_train_20250224_200802.png
│   │   ├── feature_importance_test_20250224_200818.png
│   │   ├── feature_importance_train_20250224_200802.png
│   │   ├── prediction_dist_test_20250224_200818.png
│   │   ├── prediction_dist_train_20250224_200802.png
│   │   ├── roc_curve_test_20250224_200818.png
│   │   ├── roc_curve_train_20250224_200802.png
│   │   ├── threshold_performance_test_20250224_200818.png
│   │   └── threshold_performance_train_20250224_200802.png
│   └── results/
│       ├── test_results_20250224_200818.json
│       └── train_results_20250224_200802.json
├── scripts/
│   └── generate_tree.py
├── src/
│   ├── data/
│   │   ├── data_ingestion.py
│   │   ├── data_preprocessing.py
│   │   └── initial_split.py
│   ├── evaluation/
│   │   ├── model_evaluation.py
│   │   └── test_predictor.py
│   ├── models/
│   │   ├── model_builder.py
│   │   └── model_trainer.py
│   └── utils/
│       └── logger.py
├── Dockerfile
├── ReadMe.md
├── ai_engineer_assignment.pdf
├── main.py
└── requirements.txt
```
 -->
