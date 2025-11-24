"""SHAP-based model explainability."""

import shap
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import pickle
from pathlib import Path

from ..utils import logger


class ModelExplainer:
    """Explain model predictions using SHAP values."""
    
    def __init__(self, model: Any, feature_names: List[str]):
        """
        Initialize explainer.
        
        Args:
            model: Trained model (XGBoost/LightGBM)
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.background_data = None
        
    def fit(self, background_data: pd.DataFrame, nsamples: int = 100):
        """
        Fit explainer on background data.
        
        Args:
            background_data: Sample of training data for baseline
            nsamples: Number of background samples to use
        """
        logger.info(f"Fitting SHAP explainer with {nsamples} background samples...")
        
        # Sample background data if needed
        if len(background_data) > nsamples:
            self.background_data = background_data.sample(n=nsamples, random_state=42)
        else:
            self.background_data = background_data
        
        # Create TreeExplainer for tree-based models
        self.explainer = shap.TreeExplainer(self.model)
        
        logger.info("SHAP explainer fitted successfully")
        
    def explain_prediction(
        self,
        instance: pd.DataFrame,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Explain a single prediction.
        
        Args:
            instance: Single row DataFrame to explain
            top_k: Number of top features to return
            
        Returns:
            Dictionary with explanation
        """
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call fit() first.")
        
        # Get SHAP values
        shap_values = self.explainer.shap_values(instance)
        
        # For binary classification, get positive class SHAP values
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Fraud class
        
        # Get base value
        base_value = self.explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[1]  # Fraud class
        
        # Create feature importance dict
        feature_importance = {}
        for i, feature in enumerate(self.feature_names):
            if i < len(shap_values[0]):
                feature_importance[feature] = float(shap_values[0][i])
        
        # Sort by absolute value and get top_k
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:top_k]
        
        # Get prediction value
        prediction_value = float(base_value + shap_values[0].sum())
        
        explanation = {
            'base_value': float(base_value),
            'prediction_value': prediction_value,
            'top_features': [
                {
                    'feature': feat,
                    'shap_value': value,
                    'contribution': 'increases fraud probability' if value > 0 else 'decreases fraud probability',
                    'magnitude': abs(value)
                }
                for feat, value in sorted_features
            ],
            'all_features': {feat: val for feat, val in feature_importance.items()}
        }
        
        return explanation
    
    def explain_local(
        self,
        instance: pd.DataFrame
    ) -> str:
        """
        Get human-readable explanation for a prediction.
        
        Args:
            instance: Single row DataFrame
            
        Returns:
            Human-readable explanation string
        """
        explanation = self.explain_prediction(instance, top_k=3)
        
        # Build explanation string
        parts = []
        parts.append(f"Base fraud probability: {explanation['base_value']:.2%}")
        parts.append(f"Final fraud probability: {explanation['prediction_value']:.2%}")
        parts.append("\nTop contributing factors:")
        
        for i, feat_info in enumerate(explanation['top_features'], 1):
            direction = "↑" if feat_info['shap_value'] > 0 else "↓"
            parts.append(
                f"{i}. {feat_info['feature']}: {direction} {feat_info['contribution']} "
                f"(impact: {feat_info['magnitude']:.4f})"
            )
        
        return "\n".join(parts)
