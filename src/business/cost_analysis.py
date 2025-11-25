import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

class CostCalculator:
    """
    Calculates the financial impact of the fraud detection model.
    
    This class translates technical metrics (confusion matrix) into 
    business metrics (Net Monetary Value, ROI) based on cost assumptions.
    """
    
    def __init__(
        self, 
        fixed_admin_cost: float = 10.0,
        avg_transaction_amount: float = 150.0
    ):
        """
        Initialize the cost calculator.
        
        Args:
            fixed_admin_cost: Cost to investigate a flagged transaction (False Positive cost).
            avg_transaction_amount: Average value of a transaction (used if actual amounts not provided).
        """
        self.fixed_admin_cost = fixed_admin_cost
        self.avg_transaction_amount = avg_transaction_amount
        
    def calculate_net_monetary_value(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        amounts: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate the Net Monetary Value (NMV) of the model's decisions.
        
        Logic:
        - True Positive (TP): Fraud caught. Value = +Transaction Amount (Saved).
        - False Positive (FP): False alarm. Value = -Admin Cost (Investigation).
        - False Negative (FN): Fraud missed. Value = -Transaction Amount (Lost).
        - True Negative (TN): Legitimate transaction allowed. Value = 0 (Status quo).
        
        Args:
            y_true: Ground truth labels (1=Fraud, 0=Normal).
            y_pred: Model predictions (1=Fraud, 0=Normal).
            amounts: Array of transaction amounts. If None, uses avg_transaction_amount.
            
        Returns:
            Dictionary containing financial metrics.
        """
        if amounts is None:
            amounts = np.full_like(y_true, self.avg_transaction_amount, dtype=float)
            
        # Ensure inputs are numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        amounts = np.array(amounts)
        
        # Identify confusion matrix components
        tp_mask = (y_true == 1) & (y_pred == 1)
        fp_mask = (y_true == 0) & (y_pred == 1)
        fn_mask = (y_true == 1) & (y_pred == 0)
        
        # Calculate financial impact
        fraud_saved = np.sum(amounts[tp_mask])
        investigation_cost = np.sum(fp_mask) * self.fixed_admin_cost
        fraud_lost = np.sum(amounts[fn_mask])
        
        # Net Monetary Value = Money Saved - Costs Incurred
        # Note: We don't subtract fraud_lost from NMV because NMV represents the *value added* by the model 
        # compared to doing nothing. But often businesses look at "Total Cost of Fraud" vs "Savings".
        # Let's define NMV as: Savings - Investigation Costs.
        nmv = fraud_saved - investigation_cost
        
        # Total Potential Fraud Loss (if no model existed)
        total_potential_loss = np.sum(amounts[y_true == 1])
        
        return {
            "fraud_saved": float(fraud_saved),
            "investigation_cost": float(investigation_cost),
            "fraud_lost": float(fraud_lost),
            "net_monetary_value": float(nmv),
            "total_potential_loss": float(total_potential_loss),
            "roi_percentage": float((nmv / investigation_cost * 100) if investigation_cost > 0 else 0.0)
        }
        
    def compare_thresholds(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        amounts: Optional[np.ndarray] = None,
        thresholds: List[float] = [0.1, 0.3, 0.5, 0.7, 0.9]
    ) -> pd.DataFrame:
        """
        Compare financial performance across different probability thresholds.
        """
        results = []
        
        for thresh in thresholds:
            y_pred = (y_prob >= thresh).astype(int)
            metrics = self.calculate_net_monetary_value(y_true, y_pred, amounts)
            metrics['threshold'] = thresh
            results.append(metrics)
            
        return pd.DataFrame(results)
