from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path
import numpy as np
import time
from typing import List, Dict
from src.utils.logger import setup_logger

logger = setup_logger()

app = FastAPI(
    title="DDoS Detection API",
    description="API for real-time DDoS attack detection",
    version="1.0.0"
)

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

@app.get("/")
async def root():
    """Basic health check."""
    return {"status": "API is running"}

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