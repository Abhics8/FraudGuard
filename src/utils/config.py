"""Configuration management for RiskLens fraud detection system."""

import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "fraud-detection"
    
    # Database
    database_url: str = "sqlite:///./fraud_detection.db"
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    api_title: str = "RiskLens API"
    api_version: str = "1.0.0"
    
    # Model
    model_type: str = "xgboost"
    threshold: float = 0.5
    
    # Monitoring
    drift_threshold: float = 0.05
    enable_monitoring: bool = True
    
    # Paths
    data_path: Path = Path("./data")
    model_path: Path = Path("./models")
    mlruns_path: Path = Path("./mlruns")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings."""
    return settings
