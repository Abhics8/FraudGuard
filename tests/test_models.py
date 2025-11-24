"""Unit tests for models."""

import pytest
import pandas as pd
import numpy as np
from src.models import FraudDetector


@pytest.fixture
def sample_data():
    """Create sample training data."""
    np.random.seed(42)
    n_samples = 1000
    
    # Create imbalanced dataset
    X = pd.DataFrame({
        f'V{i}': np.random.randn(n_samples) for i in range(1, 29)
    })
    X['Amount'] = np.random.exponential(100, n_samples)
    X['Time'] = np.random.randint(0, 172800, n_samples)
    
    # 5% fraud
    y = pd.Series([0] * 950 + [1] * 50)
    
    return X, y


def test_fraud_detector_initialization():
    """Test detector initialization."""
    detector = FraudDetector(model_type="xgboost", use_smote=True)
    
    assert detector.model_type == "xgboost"
    assert detector.use_smote is True
    assert detector.model is None


def test_fraud_detector_train_xgboost(sample_data):
    """Test training XGBoost model."""
    X, y = sample_data
    
    detector = FraudDetector(model_type="xgboost", use_smote=False)
    
    # Train with small params
    params = {
        'n_estimators': 10,
        'max_depth': 3,
        'learning_rate': 0.1
    }
    
    metrics = detector.train(X, y, params=params, experiment_name="test")
    
    # Check model is trained
    assert detector.model is not None
    
    # Check metrics exist
    assert 'train_precision' in metrics
    assert 'train_recall' in metrics
    assert 'train_f1' in metrics
    
    # Check metrics are reasonable
    assert 0 <= metrics['train_precision'] <= 1
    assert 0 <= metrics['train_recall'] <= 1


def test_fraud_detector_predict(sample_data):
    """Test predictions."""
    X, y = sample_data
    
    detector = FraudDetector(model_type="xgboost", use_smote=False)
    
    # Train quickly
    params = {'n_estimators': 5, 'max_depth': 2}
    detector.train(X, y, params=params, experiment_name="test")
    
    # Get predictions
    predictions = detector.predict(X.head(10), threshold=0.5)
    
    assert len(predictions) == 10
    assert all(p in [0, 1] for p in predictions)


def test_fraud_detector_predict_proba(sample_data):
    """Test probability predictions."""
    X, y = sample_data
    
    detector = FraudDetector(model_type="xgboost", use_smote=False)
    
    params = {'n_estimators': 5, 'max_depth': 2}
    detector.train(X, y, params=params, experiment_name="test")
    
    probas = detector.predict_proba(X.head(10))
    
    assert len(probas) == 10
    assert all(0 <= p <= 1 for p in probas)


def test_fraud_detector_evaluate(sample_data):
    """Test model evaluation."""
    X, y = sample_data
    
    detector = FraudDetector(model_type="xgboost", use_smote=False)
    
    params = {'n_estimators': 10, 'max_depth': 3}
    detector.train(X, y, params=params, experiment_name="test")
    
    metrics = detector.evaluate(X, y, prefix="test")
    
    # Check all expected metrics
    expected_metrics = [
        'test_precision', 'test_recall', 'test_f1',
        'test_roc_auc', 'test_pr_auc',
        'test_true_negatives', 'test_false_positives',
        'test_false_negatives', 'test_true_positives'
    ]
    
    for metric in expected_metrics:
        assert metric in metrics


def test_fraud_detector_lightgbm(sample_data):
    """Test LightGBM model."""
    X, y = sample_data
    
    detector = FraudDetector(model_type="lightgbm", use_smote=False)
    
    params = {'n_estimators': 5, 'max_depth': 2}
    metrics = detector.train(X, y, params=params, experiment_name="test")
    
    assert detector.model is not None
    assert 'train_f1' in metrics


def test_fraud_detector_invalid_model_type():
    """Test invalid model type raises error."""
    detector = FraudDetector(model_type="invalid")
    
    X = pd.DataFrame({'V1': [1, 2, 3]})
    y = pd.Series([0, 1, 0])
    
    with pytest.raises(ValueError, match="Unknown model type"):
        detector.train(X, y, experiment_name="test")
