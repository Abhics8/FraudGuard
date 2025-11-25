"""Prometheus metrics for monitoring."""

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from functools import wraps
import time
from typing import Callable

# API Metrics
api_requests_total = Counter(
    'fraudguard_api_requests_total',
    'Total number of API requests',
    ['method', 'endpoint', 'status']
)

api_request_duration_seconds = Histogram(
    'fraudguard_api_request_duration_seconds',
    'API request duration in seconds',
    ['method', 'endpoint']
)

# Prediction Metrics
predictions_total = Counter(
    'fraudguard_predictions_total',
    'Total number of predictions made',
    ['prediction_type']
)

fraud_detected_total = Counter(
    'fraudguard_fraud_detected_total',
    'Total number of fraud cases detected'
)

fraud_probability_histogram = Histogram(
    'fraudguard_fraud_probability',
    'Distribution of fraud probabilities',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

prediction_latency_seconds = Histogram(
    'fraudguard_prediction_latency_seconds',
    'Time taken for model prediction',
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5]
)

# Model Metrics
model_loaded = Gauge(
    'fraudguard_model_loaded',
    'Whether model is loaded (1) or not (0)'
)

# Drift Metrics
data_drift_detected = Gauge(
    'fraudguard_data_drift_detected',
    'Whether data drift is detected (1) or not (0)'
)

drift_share = Gauge(
    'fraudguard_drift_share',
    'Share of drifted features'
)


def track_prediction_metrics(func: Callable) -> Callable:
    """Decorator to track prediction metrics."""
    
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = await func(*args, **kwargs)
            
            # Track prediction
            predictions_total.labels(prediction_type='single').inc()
            
            # Track fraud detection
            if result.get('is_fraud', False):
                fraud_detected_total.inc()
            
            # Track probability
            prob = result.get('fraud_probability', 0)
            fraud_probability_histogram.observe(prob)
            
            # Track latency
            latency = time.time() - start_time
            prediction_latency_seconds.observe(latency)
            
            return result
            
        except Exception as e:
            # Track failed predictions
            predictions_total.labels(prediction_type='failed').inc()
            raise e
    
    return wrapper


def track_request_metrics(func: Callable) -> Callable:
    """Decorator to track API request metrics."""
    
    @wraps(func)
    async def wrapper(request, *args, **kwargs):
        start_time = time.time()
        method = request.method
        endpoint = request.url.path
        
        try:
            response = await func(request, *args, **kwargs)
            status = response.status_code if hasattr(response, 'status_code') else 200
            
            # Track request
            api_requests_total.labels(method=method, endpoint=endpoint, status=status).inc()
            
            # Track duration
            duration = time.time() - start_time
            api_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(duration)
            
            return response
            
        except Exception as e:
            # Track failed requests
            api_requests_total.labels(method=method, endpoint=endpoint, status=500).inc()
            raise e
    
    return wrapper


def get_metrics() -> bytes:
    """Get current metrics in Prometheus format."""
    return generate_latest()


def update_model_status(loaded: bool):
    """Update model loaded status."""
    model_loaded.set(1 if loaded else 0)


def update_drift_metrics(drift_detected: bool, share: float = 0.0):
    """Update drift detection metrics."""
    data_drift_detected.set(1 if drift_detected else 0)
    drift_share.set(share)
