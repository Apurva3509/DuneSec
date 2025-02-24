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
├── config/ # Configuration files
├── data/ # Data management
│ ├── raw/ # Raw network flow data
│ ├── processed/ # Preprocessed data
│ └── test/ # Hold-out test set (10%)
├── models/ # Model artifacts
│ ├── model.joblib # Trained model
│ └── preprocessors/# Feature preprocessors
├── src/ # Source code
│ ├── data/ # Data processing modules
│ ├── models/ # Model training modules
│ ├── evaluation/ # Evaluation modules
│ └── utils/ # Utility functions
├── reports/ # Performance reports
│ └── figures/ # Visualizations
└── logs/ # Application logs


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


2. **Prepare Data**
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