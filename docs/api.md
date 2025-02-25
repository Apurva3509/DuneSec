# DDoS Detection API Documentation

## Overview
The DDoS Detection API provides real-time network traffic classification using our trained Random Forest model.

## Getting Started

### Prerequisites
- Python 3.9+
- FastAPI
- uvicorn
- Required model artifacts in `models/` directory

### Starting the API Server

1. **Ensure Model Artifacts Exist**:
```bash
models/
├── random_forest_model.joblib
└── preprocessors/
    ├── scaler.joblib
    ├── label_encoder.joblib
    └── feature_names.joblib
```

2. **Start the Server**:
```bash
# Start with reload (development)
uvicorn src.api.app:app --reload --port 8000

# Start without reload (production)
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### Health Check
```
GET /
```
Checks if the API and model are running correctly.

**Response**:
```json
{
    "status": "healthy",
    "model_loaded": true
}
```

### Make Prediction
```
POST /predict
```
Classifies a network flow as either benign or DDoS attack.

**Request Body**:
```json
{
    "features": {
        "Flow Duration": 123.45,
        "Total Fwd Packets": 10,
        "Total Backward Packets": 5,
        "Flow Bytes/s": 1000.0,
        "Flow Packets/s": 50.0
        // ... other features
    }
}
```

**Response**:
```json
{
    "prediction": 1,
    "probability": 0.98,
    "prediction_time_ms": 5.23,
    "is_attack": true
}
```

## Using the API

### 1. Python Client
```python
import requests

def predict_flow(features):
    response = requests.post(
        "http://localhost:8000/predict",
        json={"features": features}
    )
    return response.json()
```

### 2. Interactive Testing
Use our test script to try predictions:
```bash
python src/api/test_prediction.py
```

### 3. Swagger UI
Access interactive API documentation at:
```
http://localhost:8000/docs
```

## Error Handling

The API returns standard HTTP status codes:
- 200: Successful prediction
- 400: Invalid input
- 500: Server error

Error responses include a detail message:
```json
{
    "detail": "Missing required features: ['Flow Duration']"
}
```

## Production Deployment

### Using Docker
```bash
# Build image
docker build -t ddos-detection-api .

# Run container
docker run -p 8000:8000 ddos-detection-api
``` 