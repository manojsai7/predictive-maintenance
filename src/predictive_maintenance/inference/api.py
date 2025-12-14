"""FastAPI inference service for predictive maintenance models."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from pathlib import Path


app = FastAPI(
    title="Predictive Maintenance Inference API",
    description="REST API for equipment health monitoring and failure prediction",
    version="0.1.0"
)


class SensorReading(BaseModel):
    """Schema for a single sensor reading."""
    timestamp: str
    sensor_values: Dict[str, float] = Field(..., description="Dictionary of sensor names to values")
    unit_id: Optional[int] = Field(None, description="Equipment unit ID")
    cycle: Optional[int] = Field(None, description="Operating cycle number")


class BatchSensorReadings(BaseModel):
    """Schema for batch sensor readings."""
    readings: List[SensorReading]


class PredictionResponse(BaseModel):
    """Schema for prediction response."""
    unit_id: Optional[int]
    predicted_rul: float = Field(..., description="Predicted Remaining Useful Life")
    confidence: Optional[float] = Field(None, description="Prediction confidence score")
    risk_level: str = Field(..., description="Risk level: low, medium, high, critical")


class HealthCheckResponse(BaseModel):
    """Schema for health check response."""
    status: str
    version: str
    models_loaded: bool


class ModelInfo(BaseModel):
    """Schema for model information."""
    model_type: str
    version: str
    last_updated: str
    metrics: Optional[Dict[str, float]]


# Global model storage (in production, use proper model registry)
loaded_models = {}


@app.get("/", response_model=HealthCheckResponse)
async def root():
    """Health check endpoint."""
    return HealthCheckResponse(
        status="healthy",
        version="0.1.0",
        models_loaded=len(loaded_models) > 0
    )


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Detailed health check."""
    return HealthCheckResponse(
        status="healthy",
        version="0.1.0",
        models_loaded=len(loaded_models) > 0
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_rul(reading: SensorReading):
    """
    Predict Remaining Useful Life for a single sensor reading.
    
    Args:
        reading: Sensor reading data
        
    Returns:
        Prediction with RUL and risk level
    """
    if not loaded_models.get('rul_model'):
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert sensor values to DataFrame
        sensor_df = pd.DataFrame([reading.sensor_values])
        
        # Make prediction (placeholder logic)
        # In production, this would use the actual loaded model
        predicted_rul = np.random.uniform(10, 100)  # Placeholder
        
        # Determine risk level based on RUL
        if predicted_rul < 10:
            risk_level = "critical"
        elif predicted_rul < 30:
            risk_level = "high"
        elif predicted_rul < 60:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        return PredictionResponse(
            unit_id=reading.unit_id,
            predicted_rul=predicted_rul,
            confidence=0.85,
            risk_level=risk_level
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(batch: BatchSensorReadings):
    """
    Predict RUL for batch of sensor readings.
    
    Args:
        batch: Batch of sensor readings
        
    Returns:
        List of predictions
    """
    if not loaded_models.get('rul_model'):
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    predictions = []
    
    for reading in batch.readings:
        try:
            pred = await predict_rul(reading)
            predictions.append(pred)
        except Exception as e:
            # Continue with other predictions even if one fails
            predictions.append(PredictionResponse(
                unit_id=reading.unit_id,
                predicted_rul=0.0,
                risk_level="unknown"
            ))
    
    return predictions


@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about loaded model."""
    if not loaded_models.get('rul_model'):
        raise HTTPException(status_code=404, detail="No model loaded")
    
    return ModelInfo(
        model_type="random_forest",
        version="0.1.0",
        last_updated="2024-01-01T00:00:00",
        metrics={"rmse": 10.5, "mae": 8.2, "r2": 0.85}
    )


@app.post("/model/load")
async def load_model(model_path: str):
    """
    Load a trained model from disk.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Success message
    """
    try:
        # In production, implement actual model loading
        # from predictive_maintenance.models.classical_models import RULPredictor
        # model = RULPredictor()
        # model.load_model(model_path)
        # loaded_models['rul_model'] = model
        
        loaded_models['rul_model'] = {"loaded": True, "path": model_path}
        
        return {"status": "success", "message": f"Model loaded from {model_path}"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@app.get("/models/list")
async def list_models():
    """List all available models."""
    return {
        "models": list(loaded_models.keys()),
        "count": len(loaded_models)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
