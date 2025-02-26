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
9. [References](#references)

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

# Step 3: Use the model on test set via WebPage
python main.py --mode serve

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
3. ['Technical Report'](docs/Report-Apurva.pdf).
---

### Modeling Strategy
The chosen model for detecting DDoS attacks is **Random Forest**, due to:
- **High interpretability**: Feature importance tracking.
- **Robustness to noisy data**.
- **Fast inference speed** (~5.3ms per prediction).

## [Implementation Details](docs/NOTES.md)

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
- Detailed API docs: [API Documentation](docs/api.md)
- Swagger UI: http://localhost:8000/docs

---
## Web Interface for DDoS Detection Demo

### Prerequisites
```bash
# Install required packages
pip install fastapi uvicorn aiofiles matplotlib seaborn
```

### Steps to Run the Web Interface

1. **Setup Directory Structure**
```bash
# Create necessary directories
mkdir -p src/api/static
```

2. **Copy Background Image**
```bash
# Copy background image to static folder
cp docs/webpage-bg.png src/api/static/
```

3. **Run the Web Server**
```bash
# Start the FastAPI server
python main.py --mode serve
```

4. **Access the Webpage**
- Open your web browser
- Go to: http://localhost:8000
- Enter an index number (0-1000) to test different samples
- View predictions, confidence scores, and feature visualizations

### Features
- Real-time predictions
- Confidence score visualization
- Top 5 important features display
- Feature value plots
- Inference time measurement
- Responsive design

### Troubleshooting
- If you get a "Module not found" error, ensure all required packages are installed
- If the background image doesn't load, verify the image path in static folder
- If the server fails to start, ensure port 8000 is not in use


---

## **View Results**
- **Model artifacts**: `models/`
- **Performance metrics**: `reports/results/`
- **Visualizations**: `reports/figures/`
- **Logs**: `logs/app.log`

---

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
- **Logging** (`logs/app.log`)

---

## Documentation
- 🔗 [**Assesment Presentation**](docs/Presentation.pdf) - **Slides for the assesment presentation**
- 🔗 [**Technical Report**](docs/Report-Apurva.pdf) - **Technical report for the assesment**
- 🔗 [**Analysis & Insights**](docs/NOTES.md) - **EDA findings and design decisions**
- 📍 [**Model Architecture**](docs/model.md) - **Model structure & training pipeline**

---

## Future Improvements
- Implement **XGBoost for better performance**.
- Develop a **real-time API using FastAPI or Flask**.
- Improve **feature engineering using domain knowledge**.

---


## References:
- https://pmc.ncbi.nlm.nih.gov/articles/PMC10578588/pdf/pone.0286652.pdf
- https://www.mdpi.com/2076-3417/11/22/10609
-  https://www.unb.ca/cic/datasets/ddos-2019.html
- https://www.unb.ca/cic/research/applications.html#CICFlowMeter

_Developed by [Apurva Patel](https://www.patelapurva.com)_

