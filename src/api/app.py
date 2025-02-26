from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
import time
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from typing import List, Dict
from src.utils.logger import setup_logger

logger = setup_logger()

app = FastAPI(
    title="DDoS Detection API",
    description="API for real-time DDoS attack detection",
    version="1.0.0"
)

# Load test data
test_data = pd.read_csv("data/processed/test.csv")
X_test = test_data.drop('Label', axis=1)
y_test = test_data['Label']
model = joblib.load("models/random_forest_model.joblib")

# Get feature importance once
feature_importance = pd.DataFrame({
    'feature': X_test.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# Mount static files
app.mount("/static", StaticFiles(directory="src/api/static"), name="static")

class NetworkFlow(BaseModel):
    features: Dict[str, float]

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    prediction_time_ms: float
    is_attack: bool

class ModelService:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.load_artifacts()

    def load_artifacts(self):
        """Load model and preprocessor artifacts."""
        try:
            # Check multiple possible locations for model artifacts
            model_path = Path("models/random_forest_model.joblib")
            scaler_path = Path("models/preprocessors/scaler.joblib")
            features_path = Path("models/preprocessors/feature_names.joblib")
            
            # Load model and preprocessors
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found at {model_path}")
            if not scaler_path.exists():
                raise FileNotFoundError(f"Scaler not found at {scaler_path}")
                
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            
            # Try to load feature names, or get from training data columns
            try:
                self.feature_names = joblib.load(features_path)
            except:
                # Fallback to default features if file doesn't exist
                self.feature_names = [
                    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
                    'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
                    'Fwd Packet Length Max', 'Fwd Packet Length Min',
                    'Flow Bytes/s', 'Flow Packets/s'
                    # Add all your features here
                ]
            
            logger.info(f"Loaded model and preprocessors successfully")
            logger.info(f"Number of features: {len(self.feature_names)}")
            
        except Exception as e:
            logger.error(f"Error loading model artifacts: {str(e)}")
            raise

    def predict(self, features: Dict[str, float]) -> Dict:
        """Make prediction on single network flow."""
        try:
            # Convert features to DataFrame with correct column order
            df = pd.DataFrame([features])
            
            # Ensure all required features are present
            missing_features = set(self.feature_names) - set(df.columns)
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
            
            # Reorder columns to match training data
            df = df[self.feature_names]
            
            # Scale features
            X_scaled = self.scaler.transform(df)
            
            # Make prediction
            start_time = time.time()
            prediction = self.model.predict(X_scaled)[0]
            probability = self.model.predict_proba(X_scaled)[0][1]
            pred_time = (time.time() - start_time) * 1000
            
            return {
                "prediction": int(prediction),
                "probability": float(probability),
                "prediction_time_ms": float(pred_time),
                "is_attack": bool(prediction == 1)
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise

# Initialize model service
model_service = ModelService()

@app.get("/", response_class=HTMLResponse)
async def root():
    with open('src/api/static/index.html', 'r') as f:
        return f.read()

@app.get("/banner")
async def get_banner():
    return FileResponse("src/ddos.png")

def create_feature_plot(features):
    plt.figure(figsize=(10, 4))
    plt.bar(features.keys(), features.values())
    plt.xticks(rotation=45, ha='right')
    plt.title('Feature Values')
    plt.tight_layout()
    
    # Convert plot to base64 string
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

@app.get("/health")
async def health_check():
    """Detailed health check."""
    try:
        model_path = Path("models/random_forest_model.joblib")
        scaler_path = Path("models/preprocessors/scaler.joblib")
        
        model_exists = model_path.exists()
        scaler_exists = scaler_path.exists()
        
        return {
            "status": "healthy",
            "model_file_exists": model_exists,
            "scaler_file_exists": scaler_exists,
            "model_path": str(model_path),
            "scaler_path": str(scaler_path)
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "unhealthy", "error": str(e)}

@app.post("/predict", response_model=PredictionResponse)
def predict(flow: NetworkFlow):
    """Make prediction on network flow."""
    try:
        result = model_service.predict(flow.features)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict/{index}")
async def predict_sample(index: int):
    try:
        # Get sample from test data
        X_sample = X_test.iloc[index]
        true_label = y_test.iloc[index]
        
        # Measure inference time
        start_time = time.time()
        
        # Make prediction
        X_sample_reshaped = X_sample.values.reshape(1, -1)
        prediction = model.predict(X_sample_reshaped)[0]
        probabilities = model.predict_proba(X_sample_reshaped)[0]
        
        # Calculate inference time
        inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Get probability for the predicted class
        prob = probabilities[1] if prediction == 1 else probabilities[0]
        
        # Get top 5 most important features for this sample
        top_features = {}
        for feature in feature_importance['feature'][:5]:
            top_features[feature] = float(X_sample[feature])
        
        # Create feature plot
        feature_plot = create_feature_plot(top_features)
        
        return {
            "index": index,
            "prediction": int(prediction),
            "true_label": int(true_label),
            "probability": float(prob),
            "features": top_features,
            "feature_plot": feature_plot,
            "inference_time": round(inference_time, 2)
        }
    except IndexError:
        raise HTTPException(status_code=404, detail=f"Index {index} out of bounds") 