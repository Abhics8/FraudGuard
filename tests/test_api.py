"""Unit tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app


client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint."""
    response = client.get("/")
    
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "docs" in data


def test_health_endpoint():
    """Test health check endpoint."""
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert data["status"] == "healthy"


def test_predict_endpoint_structure():
    """Test predict endpoint accepts valid request."""
    transaction = {
        "Time": 12000,
        "V1": -1.35, "V2": -0.07, "V3": 2.53,
        "V4": 1.38, "V5": -0.33, "V6": 0.46,
        "V7": 0.24, "V8": 0.09, "V9": 0.36,
        "V10": 0.09, "V11": -0.55, "V12": -0.62,
        "V13": -0.99, "V14": -0.31, "V15": 1.47,
        "V16": -0.47, "V17": 0.21, "V18": 0.03,
        "V19": 0.40, "V20": 0.25, "V21": -0.02,
        "V22": 0.28, "V23": -0.11, "V24": 0.07,
        "V25": 0.13, "V26": -0.19, "V27": 0.13,
        "V28": -0.02, "Amount": 149.62
    }
    
    # Will return 503 if model not loaded, but validates request structure
    response = client.post("/predict", json=transaction)
    
    # Either success or model not loaded
    assert response.status_code in [200, 503]


def test_predict_endpoint_invalid_data():
    """Test predict endpoint rejects invalid data."""
    invalid_transaction = {
        "Time": "invalid",  # Should be float
        "V1": -1.35,
        "Amount": 100.0
    }
    
    response = client.post("/predict", json=invalid_transaction)
    
    # Should return validation error
    assert response.status_code == 422


def test_predict_endpoint_missing_fields():
    """Test predict endpoint requires all fields."""
    incomplete_transaction = {
        "Time": 12000,
        "V1": -1.35
        # Missing other required fields
    }
    
    response = client.post("/predict", json=incomplete_transaction)
    
    assert response.status_code == 422


def test_batch_predict_endpoint():
    """Test batch prediction endpoint."""
    transactions = [
        {
            "Time": 12000,
            "V1": -1.35, "V2": -0.07, "V3": 2.53,
            "V4": 1.38, "V5": -0.33, "V6": 0.46,
            "V7": 0.24, "V8": 0.09, "V9": 0.36,
            "V10": 0.09, "V11": -0.55, "V12": -0.62,
            "V13": -0.99, "V14": -0.31, "V15": 1.47,
            "V16": -0.47, "V17": 0.21, "V18": 0.03,
            "V19": 0.40, "V20": 0.25, "V21": -0.02,
            "V22": 0.28, "V23": -0.11, "V24": 0.07,
            "V25": 0.13, "V26": -0.19, "V27": 0.13,
            "V28": -0.02, "Amount": 149.62
        },
        {
            "Time": 15000,
            "V1": 1.19, "V2": 0.27, "V3": 0.16,
            "V4": 0.45, "V5": 0.06, "V6": -0.08,
            "V7": -0.08, "V8": 0.08, "V9": -0.26,
            "V10": -0.17, "V11": 1.61, "V12": 1.07,
            "V13": 0.49, "V14": -0.14, "V15": 0.64,
            "V16": -0.47, "V17": 0.71, "V18": -0.17,
            "V19": -0.33, "V20": -0.19, "V21": -0.35,
            "V22": -0.77, "V23": -0.12, "V24": 0.13,
            "V25": -0.21, "V26": 0.50, "V27": 0.22,
            "V28": 0.06, "Amount": 22.50
        }
    ]
    
    response = client.post("/predict/batch", json=transactions)
    
    # Either success or model not loaded
    assert response.status_code in [200, 503]


def test_model_info_endpoint():
    """Test model info endpoint."""
    response = client.get("/model/info")
    
    # Either returns info or model not loaded
    assert response.status_code in [200, 503]


def test_openapi_docs():
    """Test OpenAPI documentation is available."""
    response = client.get("/openapi.json")
    
    assert response.status_code == 200
    schema = response.json()
    assert "openapi" in schema
    assert "paths" in schema
    
    # Check our endpoints are documented
    assert "/predict" in schema["paths"]
    assert "/health" in schema["paths"]
    assert "/model/info" in schema["paths"]
