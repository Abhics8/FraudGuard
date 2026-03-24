import pytest
from app import predict_fraud

def test_predict_fraud_low_risk():
    """Test inference heuristic for a low-risk transaction."""
    # Amount: 50, Time: 12 (noon), V1,V2,V3: 0.0
    result = predict_fraud(50.0, 12, 0.0, 0.0, 0.0)
    assert "✅ NO" in result
    assert "🟢 LOW RISK" in result

def test_predict_fraud_high_risk():
    """Test inference heuristic for a high-risk transaction."""
    # Amount: 800 (>$500), Time: 3 (Night), V1,V2,V3 = 3.0
    result = predict_fraud(800.0, 3, 3.0, 0.0, 0.0)
    assert "⚠️ YES" in result
    assert "🔴 HIGH RISK" in result

def test_predict_fraud_medium_risk():
    """Test inference heuristic for a medium-risk transaction."""
    # Amount: 250, Time: 12 (noon), V1 = 2.5
    result = predict_fraud(250.0, 12, 2.5, 0.0, 0.0)
    assert "🟡 MEDIUM RISK" in result
