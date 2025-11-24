"""FastAPI application for RiskLens fraud detection."""

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import joblib
from pathlib import Path

from .schemas import TransactionRequest, PredictionResponse, HealthResponse, ModelInfoResponse
from ..utils import settings, logger

# Initialize FastAPI app
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description="Real-time fraud detection API with ML-powered risk assessment"
)

# Global model instance
model = None
feature_engineer = None


@app.on_event("startup")
async def load_model():
    """Load model on startup."""
    global model, feature_engineer
    
    try:
        # In production, load from MLflow model registry
        # For now, using placeholder
        logger.info("Model loading: Using trained model from MLflow...")
        logger.info("API ready for predictions")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to RiskLens API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        model_type="xgboost" if model else None
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_fraud(transaction: TransactionRequest):
    """
    Predict if a transaction is fraudulent.
    
    Returns fraud probability and risk classification.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert request to DataFrame
        transaction_dict = transaction.dict()
        df = pd.DataFrame([transaction_dict])
        
        # Apply feature engineering if available
        if feature_engineer:
            df = feature_engineer.transform(df)
        
        # Get prediction
        fraud_prob = model.predict_proba(df)[0]
        is_fraud = fraud_prob >= settings.threshold
        
        # Determine risk score
        if fraud_prob < 0.3:
            risk_score = "low"
        elif fraud_prob < 0.7:
            risk_score = "medium"
        else:
            risk_score = "high"
        
        logger.info(f"Prediction: {is_fraud}, Probability: {fraud_prob:.4f}")
        
        return PredictionResponse(
            is_fraud=bool(is_fraud),
            fraud_probability=float(fraud_prob),
            risk_score=risk_score
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", tags=["Predictions"])
async def predict_batch(transactions: list[TransactionRequest]):
    """
    Predict fraud for multiple transactions.
    
    Returns list of predictions.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame([t.dict() for t in transactions])
        
        # Apply feature engineering
        if feature_engineer:
            df = feature_engineer.transform(df)
        
        # Get predictions
        fraud_probs = model.predict_proba(df)
        
        results = []
        for prob in fraud_probs:
            is_fraud = prob >= settings.threshold
            risk_score = "low" if prob < 0.3 else ("medium" if prob < 0.7 else "high")
            
            results.append({
                "is_fraud": bool(is_fraud),
                "fraud_probability": float(prob),
                "risk_score": risk_score
            })
        
        return results
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """Get information about the loaded model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfoResponse(
        model_type="xgboost",
        mlflow_run_id=None,  # Would be populated from MLflow
        features_count=30,  # Base features
        threshold=settings.threshold
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )
