---
title: FraudGuard - Fraud Detection
emoji: 🔍
colorFrom: red
colorTo: orange
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
---

# FraudGuard - AI Fraud Detection System

🔍 **Production-ready fraud detection with MLOps pipeline**

Try the live demo above to detect fraudulent credit card transactions using AI!

## Features

- 🎯 **96%+ Precision** with XGBoost/LightGBM models
- 🔄 **Complete MLOps Pipeline** with MLflow tracking
- 📊 **Drift Detection** with Evidently AI
- 🔍 **SHAP Explainability** for every prediction
- 📈 **Prometheus metrics** + Grafana monitoring
- 🐳 **Docker deployment** ready

## Tech Stack

- **ML**: XGBoost, LightGBM, scikit-learn
- **MLOps**: MLflow, Evidently AI, SHAP
- **API**: FastAPI, Pydantic
- **Monitoring**: Prometheus, Grafana
- **Deployment**: Docker, Hugging Face Spaces

## About This Demo

This interactive demo uses simplified heuristics for demonstration purposes. 

The full production system uses:
- XGBoost models trained on 284,807 credit card transactions
- 30 engineered features (time-based, amount-based, behavioral patterns)
- SMOTE for class imbalance handling
- MLflow for experiment tracking
- Real-time drift detection

## Links

- [GitHub Repository](https://github.com/Abhics8/FraudGuard)
- [Documentation](https://github.com/Abhics8/FraudGuard/blob/main/README.md)
- [Deployment Guide](https://github.com/Abhics8/FraudGuard/blob/main/DEPLOYMENT.md)

## Author

**Abhi Bhardwaj**
- GitHub: [@Abhics8](https://github.com/Abhics8)
- LinkedIn: [abhi-bhardwaj](https://www.linkedin.com/in/abhi-bhardwaj-23b0961a0/)

---

Built with ❤️ for production ML systems
