import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.business.cost_analysis import CostCalculator

# Page Config
st.set_page_config(
    page_title="FraudGuard Business Dashboard",
    page_icon="💰",
    layout="wide"
)

# Title
st.title("💰 FraudGuard: Business Impact Analysis")
st.markdown("""
This dashboard visualizes the **financial impact** of the FraudGuard fraud detection model.
Adjust the cost parameters in the sidebar to see how they affect Net Monetary Value (NMV) and ROI.
""")

# Sidebar: Cost Parameters
st.sidebar.header("⚙️ Cost Assumptions")
avg_transaction = st.sidebar.number_input(
    "Average Transaction Value ($)", 
    min_value=10.0, 
    max_value=1000.0, 
    value=150.0,
    step=10.0
)

admin_cost = st.sidebar.number_input(
    "Admin Investigation Cost ($)", 
    min_value=1.0, 
    max_value=100.0, 
    value=10.0,
    step=1.0
)

fraud_rate = st.sidebar.slider(
    "Estimated Fraud Rate (%)",
    min_value=0.1,
    max_value=5.0,
    value=0.5,
    step=0.1
) / 100

# Simulation Data
# In a real app, we would load test set predictions. 
# Here we simulate a realistic distribution for demonstration.
np.random.seed(42)
n_samples = 10000
n_fraud = int(n_samples * fraud_rate)
n_normal = n_samples - n_fraud

# Simulate scores: Fraud has higher scores, Normal has lower
fraud_scores = np.random.beta(5, 2, n_fraud)  # Skewed towards 1
normal_scores = np.random.beta(1, 5, n_normal) # Skewed towards 0

y_true = np.concatenate([np.ones(n_fraud), np.zeros(n_normal)])
y_scores = np.concatenate([fraud_scores, normal_scores])
amounts = np.random.exponential(avg_transaction, n_samples)

# Initialize Calculator
calculator = CostCalculator(fixed_admin_cost=admin_cost, avg_transaction_amount=avg_transaction)

# Analysis
thresholds = np.linspace(0.0, 1.0, 50)
results = calculator.compare_thresholds(y_true, y_scores, amounts, thresholds)

# Optimal Threshold
best_row = results.loc[results['net_monetary_value'].idxmax()]
optimal_threshold = best_row['threshold']
max_savings = best_row['net_monetary_value']

# --- Layout ---

# Top Metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Optimal Threshold", f"{optimal_threshold:.2f}")
with col2:
    st.metric("Max Net Savings", f"${max_savings:,.0f}")
with col3:
    st.metric("ROI", f"{best_row['roi_percentage']:.1f}%")
with col4:
    st.metric("Fraud Caught", f"${best_row['fraud_saved']:,.0f}")

# Charts
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("💸 Financial Impact vs. Threshold")
    
    fig = go.Figure()
    
    # NMV Curve
    fig.add_trace(go.Scatter(
        x=results['threshold'], 
        y=results['net_monetary_value'],
        mode='lines',
        name='Net Monetary Value',
        line=dict(color='green', width=3)
    ))
    
    # Fraud Lost Curve
    fig.add_trace(go.Scatter(
        x=results['threshold'], 
        y=results['fraud_lost'],
        mode='lines',
        name='Fraud Loss (Missed)',
        line=dict(color='red', dash='dash')
    ))
    
    # Investigation Cost Curve
    fig.add_trace(go.Scatter(
        x=results['threshold'], 
        y=results['investigation_cost'],
        mode='lines',
        name='Investigation Cost',
        line=dict(color='orange', dash='dot')
    ))
    
    fig.add_vline(x=optimal_threshold, line_dash="dash", line_color="green", annotation_text="Optimal")
    
    fig.update_layout(
        xaxis_title="Model Probability Threshold",
        yaxis_title="USD ($)",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("📊 Model Performance")
    
    # Confusion Matrix at Optimal Threshold
    y_pred_opt = (y_scores >= optimal_threshold).astype(int)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred_opt)
    
    # Custom Heatmap
    z = cm[::-1] # Flip for standard layout (TP top left)
    x = ['Normal', 'Fraud']
    y = ['Fraud', 'Normal']
    
    fig_cm = px.imshow(z, x=x, y=y, color_continuous_scale='Blues', text_auto=True)
    fig_cm.update_layout(title=f"Confusion Matrix (Thresh={optimal_threshold:.2f})")
    st.plotly_chart(fig_cm, use_container_width=True)
    
    st.info(f"""
    **Interpretation:**
    At the optimal threshold of **{optimal_threshold:.2f}**, the model maximizes profit by balancing the cost of missing fraud (FN) against the cost of disturbing customers (FP).
    """)

# Raw Data
with st.expander("See Detailed Analysis Data"):
    st.dataframe(results.style.format({
        'net_monetary_value': '${:,.2f}',
        'fraud_saved': '${:,.2f}',
        'investigation_cost': '${:,.2f}',
        'roi_percentage': '{:.1f}%'
    }))
