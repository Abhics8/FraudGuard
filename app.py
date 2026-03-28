"""Gradio interface for FraudGuard fraud detection."""

import gradio as gr
import pandas as pd
import numpy as np
from pathlib import Path

# Simple demo without requiring actual model
def predict_fraud(amount, time_of_day, v1, v2, v3):
    """
    Predict fraud probability for a transaction.
    
    This is a demo version using heuristics.
    Replace with actual model loading in production.
    """
    # Demo logic - replace with actual model
    risk_score = 0.0
    
    # High amount increases risk
    if amount > 500:
        risk_score += 0.3
    elif amount > 200:
        risk_score += 0.15
    
    # Night time increases risk
    if time_of_day < 6 or time_of_day > 22:
        risk_score += 0.2
    
    # Anomalous PCA values
    if abs(v1) > 2 or abs(v2) > 2 or abs(v3) > 2:
        risk_score += 0.25
    
    # Cap at 1.0
    fraud_probability = min(risk_score, 0.95)
    
    # Determine risk level
    if fraud_probability < 0.3:
        risk_level = "🟢 LOW RISK"
        color = "green"
    elif fraud_probability < 0.7:
        risk_level = "🟡 MEDIUM RISK"
        color = "orange"
    else:
        risk_level = "🔴 HIGH RISK"
        color = "red"
    
    is_fraud = fraud_probability >= 0.5
    
    # Format output
    result = f"""
    ### Prediction Results
    
    **Fraud Detected:** {'⚠️ YES' if is_fraud else '✅ NO'}
    
    **Fraud Probability:** {fraud_probability:.2%}
    
    **Risk Level:** {risk_level}
    
    ---
    
    ### Top Contributing Factors
    
    1. **Transaction Amount**: ${amount:.2f}
       - {'High amount increases risk' if amount > 200 else 'Normal amount'}
    
    2. **Time of Day**: {int(time_of_day)}:00
       - {'Night time increases risk' if (time_of_day < 6 or time_of_day > 22) else 'Normal business hours'}
    
    3. **Behavioral Patterns**: V1={v1:.2f}, V2={v2:.2f}, V3={v3:.2f}
       - {'Anomalous pattern detected' if any(abs(x) > 2 for x in [v1, v2, v3]) else 'Normal pattern'}
    
    ---
    
    💡 **Note:** This is a demonstration. In production, predictions use XGBoost models 
    trained on 284,807 credit card transactions with 96%+ precision.
    """
    
    return result


# Create Gradio interface
with gr.Blocks(title="FraudGuard - Fraud Detection") as demo:
    gr.Markdown("""
    # 🔍 FraudGuard - AI Fraud Detection System
    
    **Production-ready fraud detection with MLOps pipeline**
    
    Detect fraudulent credit card transactions using machine learning.
    This demo uses simplified heuristics - the full system uses XGBoost with 96%+ precision.
    
    [GitHub Repository](https://github.com/Abhics8/FraudGuard) | [Documentation](https://github.com/Abhics8/FraudGuard/blob/main/README.md)
    """)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Transaction Details")
            
            amount = gr.Slider(
                minimum=0,
                maximum=1000,
                value=100,
                step=0.01,
                label="Transaction Amount ($)",
                info="Enter transaction amount"
            )
            
            time_of_day = gr.Slider(
                minimum=0,
                maximum=23,
                value=14,
                step=1,
                label="Hour of Day (0-23)",
                info="What time did the transaction occur?"
            )
            
            gr.Markdown("### Behavioral Features (PCA Components)")
            gr.Markdown("*These represent anonymized transaction patterns*")
            
            v1 = gr.Slider(
                minimum=-5,
                maximum=5,
                value=0,
                step=0.1,
                label="V1 (Principal Component 1)",
                info="First behavioral pattern"
            )
            
            v2 = gr.Slider(
                minimum=-5,
                maximum=5,
                value=0,
                step=0.1,
                label="V2 (Principal Component 2)",
                info="Second behavioral pattern"
            )
            
            v3 = gr.Slider(
                minimum=-5,
                maximum=5,
                value=0,
                step=0.1,
                label="V3 (Principal Component 3)",
                info="Third behavioral pattern"
            )
            
            predict_btn = gr.Button("🔍 Analyze Transaction", variant="primary", size="lg")
        
        with gr.Column():
            gr.Markdown("### Analysis Results")
            output = gr.Markdown(label="Prediction")
    
    # Examples
    gr.Markdown("### 📋 Try These Examples")
    gr.Examples(
        examples=[
            [25.50, 14, 0.5, -0.3, 0.1],  # Normal transaction
            [850.00, 23, 3.2, -2.8, 2.5],  # High risk
            [150.00, 3, -1.5, 2.1, -1.8],  # Medium risk
            [500.00, 12, 0.0, 0.0, 0.0],   # Borderline
        ],
        inputs=[amount, time_of_day, v1, v2, v3],
        label="Click an example to load it"
    )
    
    # Connect button
    predict_btn.click(
        fn=predict_fraud,
        inputs=[amount, time_of_day, v1, v2, v3],
        outputs=output
    )
    
    gr.Markdown("""
    ---
    
    ## 🚀 About FraudGuard
    
    FraudGuard is a production-grade fraud detection system featuring:
    - 🎯 96%+ Precision with XGBoost/LightGBM
    - 🔄 Complete MLOps Pipeline with MLflow tracking
    - 📊 Drift Detection with Evidently AI
    - 🔍 SHAP Explainability for predictions
    - 📈 Prometheus metrics + Grafana monitoring
    - 🐳 Docker deployment ready
    
    **Tech Stack:** Python, FastAPI, XGBoost, MLflow, Evidently AI, SHAP, Prometheus, Docker
    
    Built by [Abhi Bhardwaj](https://github.com/Abhics8) | [View Source Code](https://github.com/Abhics8/FraudGuard)
    """)

# Launch the app
if __name__ == "__main__":
    demo.launch()
