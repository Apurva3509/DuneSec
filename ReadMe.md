# DDoS Attack Detection System

## Overview
This project provides an enterprise-grade machine learning solution for real-time DDoS attack detection using XGBoost. The system processes network flow data to classify traffic as either benign or a DDoS attack, achieving high accuracy and low latency.

## Features
- **Real-time Classification**: Quickly identifies DDoS attacks in network traffic.
- **High Accuracy**: Achieves over 95% accuracy on test data.
- **Low Latency**: Predictions are made in under 10ms.
- **Scalable Architecture**: Designed for production environments.
- **Comprehensive Logging**: Detailed logs for monitoring and debugging.
- **Model Versioning**: Track and manage different model versions.
- **Automated Testing**: Continuous integration and testing pipeline.

## Project Structure
ddos_detection/
├── config/
│   └── config.yaml
├── data/
│   ├── processed/
│   ├── raw/
│   └── test/
├── models/
├── notebooks/
├── reports/
│   ├── figures/
│   └── results/
├── scripts/
│   └── generate_tree.py
└── src/
    ├── data/
    │   ├── __init__.py
    │   └── data_preprocessing.py
    ├── evaluation/
    │   ├── __init__.py
    │   └── model_evaluation.py
    ├── models/
    │   ├── __init__.py
    │   └── model_trainer.py
    └── utils/
        ├── __init__.py
        └── logger.py


1. **Setup Environment**
```bash
# Clone the repository
git clone https://github.com/yourusername/ddos_detection.git
cd ddos_detection

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
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
```bash
# Create necessary directories
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/test
mkdir -p models/preprocessors
mkdir -p reports/figures
mkdir -p logs

# Place your network traffic CSV file in data/raw/
# Example: Copy your data file to data/raw/network_traffic.csv
```


3. **Run the Pipeline**
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

Check Results
- Model artifacts will be in models/ directory
- Visualizations will be in reports/figures/ directory
- Logs will be in logs/ directory


5. **Monitor Performance**
```bash
# View the latest log entries
tail -f logs/app.log

# Check model performance metrics in
cat reports/test_results.txt
```
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