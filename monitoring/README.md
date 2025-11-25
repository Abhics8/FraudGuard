# Monitoring Setup Guide

## 🎯 Overview

FraudGuard includes comprehensive monitoring with **Prometheus** metrics and **Grafana** dashboards.

## 📊 Metrics Tracked

### API Metrics
- **fraudguard_api_requests_total** - Total API requests by method, endpoint, and status
- **fraudguard_api_request_duration_seconds** - Request latency histogram

### Prediction Metrics
- **fraudguard_predictions_total** - Total predictions made
- **fraudguard_fraud_detected_total** - Fraud cases detected
- **fraudguard_fraud_probability** - Distribution of fraud probabilities
- **fraudguard_prediction_latency_seconds** - Model inference latency

### Model Metrics
- **fraudguard_model_loaded** - Model status (loaded/not loaded)

### Drift Metrics
- **fraudguard_data_drift_detected** - Data drift detected (yes/no)
- **fraudguard_drift_share** - Share of drifted features

## 🚀 Quick Start

### Using Docker Compose

```bash
# Start all services (API + MLflow + Prometheus + Grafana)
docker-compose up --build

# Access services:
# - API: http://localhost:8000
# - API Docs: http://localhost:8000/docs
# - Metrics: http://localhost:8000/metrics
# - MLflow: http://localhost:5000
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000
```

### Grafana Setup

1. **Login to Grafana**
   - URL: http://localhost:3000
   - Username: `admin`
   - Password: `admin`

2. **Add Prometheus Data Source**
   - Go to Configuration > Data Sources
   - Add Prometheus
   - URL: `http://prometheus:9090`
   - Click "Save & Test"

3. **Import Dashboard**
   - Go to Dashboards > Import
   - Upload `monitoring/grafana_dashboard.json`
   - Select Prometheus data source
   - Click "Import"

## 📈 Dashboard Panels

The Grafana dashboard includes:

1. **API Requests Rate** - Requests per second by endpoint
2. **API Request Duration (p95)** - 95th percentile latency
3. **Predictions Per Minute** - Prediction throughput
4. **Fraud Detection Rate** - Fraud cases per second
5. **Fraud Probability Distribution** - Heatmap of probabilities
6. **Model Prediction Latency** - p50, p95, p99 latencies
7. **Model Status** - Model loaded indicator
8. **Data Drift Status** - Drift detection indicator
9. **Drift Share** - Percentage of drifted features

## 🔍 Monitoring in Action

###  View Raw Metrics

```bash
curl http://localhost:8000/metrics
```

### Query with Prometheus

```promql
# API request rate
rate(fraudguard_api_requests_total[5m])

# Average prediction latency
rate(fraudguard_prediction_latency_seconds_sum[5m]) / 
rate(fraudguard_prediction_latency_seconds_count[5m])

# Fraud detection percentage
rate(fraudguard_fraud_detected_total[5m]) / 
rate(fraudguard_predictions_total[5m]) * 100
```

## 🚨 Alerting (Coming Soon)

Configure Prometheus alerts for:
- High API error rate (> 5%)
- Slow predictions (p95 > 100ms)
- Data drift detected
- Model not loaded

## 📝 Custom Metrics

Add custom metrics in `src/monitoring/metrics.py`:

```python
from prometheus_client import Counter

custom_metric = Counter(
    'fraudguard_custom_metric',
    'Description of metric',
    ['label1', 'label2']
)

# Increment
custom_metric.labels(label1='value1', label2='value2').inc()
```

## 🔧 Production Deployment

For production, consider:
- **Prometheus retention**: Increase data retention period
- **Grafana auth**: Enable authentication and user management  
- **Alertmanager**: Set up alert notifications (email, Slack, PagerDuty)
- **Remote storage**: Use Thanos or Cortex for long-term storage

---

**Build production-grade monitoring for your ML system!** 🎯
