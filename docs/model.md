# Model Architecture Documentation

## Overview
The DDoS Attack Detection System uses a Random Forest Classifier as its primary model, optimized through grid search cross-validation.

## Model Architecture

### Base Model
- **Algorithm**: Random Forest Classifier
- **Implementation**: scikit-learn's RandomForestClassifier

### Hyperparameters
```python
param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [None, 10, 20],
    "class_weight": ["balanced"],
    "min_samples_split": [2, 5]
}
```

### Cross-Validation
- **Method**: StratifiedKFold
- **Folds**: 5
- **Scoring Metric**: F1-Score

## Feature Engineering

### Preprocessing Steps
1. Standard Scaling (zero mean, unit variance)
2. Label Encoding for target variable
3. Handling missing values and infinities

### Feature Selection
- Based on feature importance analysis
- Removal of highly correlated features

## Model Performance

### Metrics
- Accuracy: 99.2%
- Precision: 98.7%
- Recall: 99.5%
- F1-Score: 99.1%
- ROC-AUC: 0.998

### Latency
- Average Prediction Time: ~5.3ms
- Batch Processing Capability: 1024 samples

## Model Artifacts

### Location
```
model_artifacts/
├── random_forest_model.joblib
├── label_encoder.joblib
└── scaler.joblib
```

### Versioning
- Model versioning handled through MLflow
- Experiments tracked in `mlruns/` directory

## Training Pipeline

### Data Split
- Training Set: 80%
- Test Set: 20%
- Stratified sampling to maintain class distribution

### Training Process
1. Data preprocessing
2. Grid search cross-validation
3. Model evaluation
4. Artifact saving

## Monitoring & Maintenance

### Performance Monitoring
- Model retraining triggers

### Logging
- Model predictions
- Performance metrics
- System health metrics

## Future Improvements
1. Implementation of XGBoost for better performance
2. Feature engineering optimization
3. API for real-time predictions via Cloud
4. Model compression for faster inference
5. Feature drift detection


## References

Links I found helpful:

1. For Data:
 -  https://www.unb.ca/cic/datasets/ddos-2019.html
 - https://www.unb.ca/cic/research/applications.html#CICFlowMeter

 ## References:
 - https://pmc.ncbi.nlm.nih.gov/articles/PMC10578588/pdf/pone.0286652.pdf
 - https://www.mdpi.com/2076-3417/11/22/10609
 
- [DDoS Detection Notebook](notebooks/DDoS-detection.ipynb)
- [Model Training Code](src/models/model_trainer.py)
- [Model Evaluation](src/evaluation/model_evaluation.py) 
