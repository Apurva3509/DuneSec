# DDoS Attack Detection System

## Overview
A machine learning solution for real-time DDoS attack detection using Random Forest classification. The system processes network flow data to classify traffic as either 'Benign' or 'DDoS' attack.

## Key Features
- Real-time Classification (< 10ms latency)
- High Accuracy (> 95% on test data)
- Comprehensive Logging & Monitoring
- MLflow Integration for Experiment Tracking
- Automated Testing Pipeline
- Docker Support

## Quick Start

### 1. Setup Environment
```bash
# Clone the repository
git clone https://github.com/Apurva3509/DuneSec.git
cd DuneSec

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation
```bash
# Create necessary directories
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/test
mkdir -p models/preprocessors
mkdir -p reports/figures
mkdir -p logs

# Place your network traffic CSV file in data/raw/network_traffic.csv
```

### 3. Run the Pipeline
```bash
# Step 1: Split the data into train (90%) and test (10%) sets
python main.py --mode split

# Step 2: Train the model
python main.py --mode train

# Step 3: Evaluate the model on test set
python main.py --mode test

# Step 4: Make predictions (when you have new data)
python main.py --mode predict
```


### 4. View Results
- Model artifacts: `models/`
- Performance metrics: `reports/results/`
- Visualizations: `reports/figures/`
- Logs: `logs/app.log`


## Model Performance
- Accuracy: 99.2%
- Precision: 98.7%
- Recall: 99.5%
- F1-Score: 99.1%
- ROC-AUC: 0.998
- Average Prediction Time: 5.3ms

## Monitoring & Logging
- Real-time performance metrics
- Feature importance tracking
- Data drift detection
- Model versioning with MLflow

## Documentation
- [Analysis & Insights](NOTES.md) - Detailed EDA findings and design decisions
- [API Documentation](docs/api.md)
- [Model Architecture](docs/model.md)

## Contributing
1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

# Analysis & Design Notes

## EDA Insights
[Reference to DDoS_detection.ipynb for EDA implementation]

The system will:
- Drop highly correlated features
- Preprocess the data
- Train the XGBoost model
- Generate performance reports
- Save the model and preprocessors

You can find:
- Model performance metrics in reports/figures/
- Trained model in models/model.joblib
- Preprocessors in models/preprocessors/
- Logs in logs/app.log



## Project Structure
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
│   │   │   │   ├── cv_score_mean
│   │   │   │   └── cv_score_std
│   │   │   ├── params/
│   │   │   │   ├── eval_metric
│   │   │   │   ├── learning_rate
│   │   │   │   ├── max_depth
│   │   │   │   ├── n_estimators
│   │   │   │   └── objective
│   │   │   ├── tags/
│   │   │   │   ├── mlflow.source.git.commit
│   │   │   │   ├── mlflow.source.name
│   │   │   │   ├── mlflow.source.type
│   │   │   │   └── mlflow.user
│   │   │   └── meta.yaml
│   │   ├── 27e907663df046ed8a200124db985bc6/
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
│   │   ├── 511934cf31b04653a01169ddbd3c5bf1/
│   │   │   ├── artifacts/
│   │   │   ├── metrics/
│   │   │   │   ├── cv_score_mean
│   │   │   │   └── cv_score_std
│   │   │   ├── params/
│   │   │   │   ├── eval_metric
│   │   │   │   ├── learning_rate
│   │   │   │   ├── max_depth
│   │   │   │   ├── n_estimators
│   │   │   │   └── objective
│   │   │   ├── tags/
│   │   │   │   ├── mlflow.source.git.commit
│   │   │   │   ├── mlflow.source.name
│   │   │   │   ├── mlflow.source.type
│   │   │   │   └── mlflow.user
│   │   │   └── meta.yaml
│   │   ├── a5b66f52f27446a1b518ab7908032e80/
│   │   │   ├── artifacts/
│   │   │   ├── metrics/
│   │   │   ├── params/
│   │   │   ├── tags/
│   │   │   │   ├── mlflow.source.git.commit
│   │   │   │   ├── mlflow.source.name
│   │   │   │   ├── mlflow.source.type
│   │   │   │   └── mlflow.user
│   │   │   └── meta.yaml
│   │   ├── f55208b492c34e9cb3bf89795f6d14af/
│   │   │   ├── artifacts/
│   │   │   ├── metrics/
│   │   │   │   ├── cv_score_mean
│   │   │   │   └── cv_score_std
│   │   │   ├── params/
│   │   │   │   ├── eval_metric
│   │   │   │   ├── learning_rate
│   │   │   │   ├── max_depth
│   │   │   │   ├── n_estimators
│   │   │   │   └── objective
│   │   │   ├── tags/
│   │   │   │   ├── mlflow.source.git.commit
│   │   │   │   ├── mlflow.source.name
│   │   │   │   ├── mlflow.source.type
│   │   │   │   └── mlflow.user
│   │   │   └── meta.yaml
│   │   ├── fa8e4d6257cc419f911eaa4caa61d2a2/
│   │   │   ├── artifacts/
│   │   │   ├── metrics/
│   │   │   ├── params/
│   │   │   ├── tags/
│   │   │   │   ├── mlflow.source.git.commit
│   │   │   │   ├── mlflow.source.name
│   │   │   │   ├── mlflow.source.type
│   │   │   │   └── mlflow.user
│   │   │   └── meta.yaml
│   │   └── meta.yaml
│   └── meta.yaml
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
├── reports/
│   ├── figures/
│   │   ├── confusion_matrix_test_20250224_144439.png
│   │   ├── confusion_matrix_train_20250224_143714.png
│   │   ├── feature_importance_20250224_143714.png
│   │   ├── feature_importance_20250224_143801.png
│   │   ├── feature_importance_test_20250224_144439.png
│   │   ├── prediction_dist_test_20250224_144412.png
│   │   ├── prediction_dist_test_20250224_144439.png
│   │   ├── roc_curve_test_20250224_144439.png
│   │   ├── roc_curve_train_20250224_143714.png
│   │   ├── roc_curve_train_20250224_143801.png
│   │   └── threshold_performance_test_20250224_144439.png
│   └── results/
│       ├── test_results_20250224_144439.json
│       ├── train_results_20250224_143714.json
│       └── train_results_20250224_143801.json
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
│   ├── utils/
│   │   └── logger.py
│   └── ddos_detection.py
├── DDoS-detection.ipynb
├── Dockerfile
├── ReadMe.md
├── ai_engineer_assignment.pdf
├── main.py
└── requirements.txt
```
