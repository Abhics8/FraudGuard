"""Pydantic schemas for API requests and responses."""

from pydantic import BaseModel, Field
from typing import List, Optional


class TransactionRequest(BaseModel):
    """Request schema for fraud prediction."""
    
    Time: float = Field(..., description="Seconds elapsed since first transaction")
    V1: float = Field(..., description="PCA component 1")
    V2: float = Field(..., description="PCA component 2")
    V3: float = Field(..., description="PCA component 3")
    V4: float = Field(..., description="PCA component 4")
    V5: float = Field(..., description="PCA component 5")
    V6: float = Field(..., description="PCA component 6")
    V7: float = Field(..., description="PCA component 7")
    V8: float = Field(..., description="PCA component 8")
    V9: float = Field(..., description="PCA component 9")
    V10: float = Field(..., description="PCA component 10")
    V11: float = Field(..., description="PCA component 11")
    V12: float = Field(..., description="PCA component 12")
    V13: float = Field(..., description="PCA component 13")
    V14: float = Field(..., description="PCA component 14")
    V15: float = Field(..., description="PCA component 15")
    V16: float = Field(..., description="PCA component 16")
    V17: float = Field(..., description="PCA component 17")
    V18: float = Field(..., description="PCA component 18")
    V19: float = Field(..., description="PCA component 19")
    V20: float = Field(..., description="PCA component 20")
    V21: float = Field(..., description="PCA component 21")
    V22: float = Field(..., description="PCA component 22")
    V23: float = Field(..., description="PCA component 23")
    V24: float = Field(..., description="PCA component 24")
    V25: float = Field(..., description="PCA component 25")
    V26: float = Field(..., description="PCA component 26")
    V27: float = Field(..., description="PCA component 27")
    V28: float = Field(..., description="PCA component 28")
    Amount: float = Field(..., description="Transaction amount", ge=0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "Time": 12000,
                "V1": -1.35,
                "V2": -0.07,
                "V3": 2.53,
                "V4": 1.38,
                "V5": -0.33,
                "V6": 0.46,
                "V7": 0.24,
                "V8": 0.09,
                "V9": 0.36,
                "V10": 0.09,
                "V11": -0.55,
                "V12": -0.62,
                "V13": -0.99,
                "V14": -0.31,
                "V15": 1.47,
                "V16": -0.47,
                "V17": 0.21,
                "V18": 0.03,
                "V19": 0.40,
                "V20": 0.25,
                "V21": -0.02,
                "V22": 0.28,
                "V23": -0.11,
                "V24": 0.07,
                "V25": 0.13,
                "V26": -0.19,
                "V27": 0.13,
                "V28": -0.02,
                "Amount": 149.62
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for fraud prediction."""
    
    is_fraud: bool = Field(..., description="Whether transaction is fraudulent")
    fraud_probability: float = Field(..., description="Probability of fraud (0-1)")
    risk_score: str = Field(..., description="Risk level: low, medium, high")
    
    class Config:
        json_schema_extra = {
            "example": {
                "is_fraud": False,
                "fraud_probability": 0.023,
                "risk_score": "low"
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_type: Optional[str] = Field(None, description="Type of model")


class ModelInfoResponse(BaseModel):
    """Model information response."""
    
    model_type: str
    mlflow_run_id: Optional[str] = None
    features_count: int
    threshold: float
