"""Unit tests for data processing."""

import pytest
import pandas as pd
import numpy as np
from src.data import load_credit_card_fraud_data, split_data, get_feature_target_split, FraudFeatureEngineer


def test_get_feature_target_split():
    """Test feature/target splitting."""
    # Create sample data
    df = pd.DataFrame({
        'V1': [1, 2, 3],
        'V2': [4, 5, 6],
        'Amount': [10.0, 20.0, 30.0],
        'Class': [0, 1, 0]
    })
    
    X, y = get_feature_target_split(df, target_column='Class')
    
    assert len(X) == 3
    assert len(y) == 3
    assert 'Class' not in X.columns
    assert list(y) == [0, 1, 0]


def test_split_data():
    """Test data splitting with stratification."""
    # Create sample data with class imbalance
    df = pd.DataFrame({
        'V1': np.random.randn(1000),
        'Class': [0] * 950 + [1] * 50  # 5% fraud
    })
    
    train, val, test = split_data(df, test_size=0.2, val_size=0.1, stratify_column='Class')
    
    # Check sizes
    assert len(train) + len(val) + len(test) == 1000
    assert len(test) == pytest.approx(200, abs=10)
    
    # Check stratification (fraud rate should be similar)
    train_fraud_rate = train['Class'].mean()
    test_fraud_rate = test['Class'].mean()
    assert train_fraud_rate == pytest.approx(0.05, abs=0.02)
    assert test_fraud_rate == pytest.approx(0.05, abs=0.02)


def test_feature_engineer():
    """Test feature engineering."""
    # Create sample data
    df = pd.DataFrame({
        'Time': [3600, 7200, 82800],  # 1hr, 2hr, 23hr
        'Amount': [10.0, 100.0, 1000.0],
        'V1': [1.0, 2.0, 3.0]
    })
    
    engineer = FraudFeatureEngineer()
    
    # Fit and transform
    transformed = engineer.fit_transform(df)
    
    # Check new features exist
    assert 'Amount_Scaled' in transformed.columns
    assert 'Hour' in transformed.columns
    assert 'Is_Night' in transformed.columns
    assert 'Amount_Log' in transformed.columns
    assert 'Is_Small_Transaction' in transformed.columns
    assert 'Is_Large_Transaction' in transformed.columns
    
    # Check feature values
    assert transformed['Hour'].iloc[0] == pytest.approx(1.0, abs=0.1)
    assert transformed['Is_Night'].iloc[2] == 1  # 23hr is night
    assert transformed['Is_Small_Transaction'].iloc[0] == 1  # $10 is small
    assert transformed['Is_Large_Transaction'].iloc[2] == 1  # $1000 is large


def test_feature_engineer_transform_only():
    """Test that transform fails without fit."""
    df = pd.DataFrame({'Amount': [10.0]})
    engineer = FraudFeatureEngineer()
    
    with pytest.raises(ValueError, match="must be fitted"):
        engineer.transform(df)
