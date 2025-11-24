"""Feature engineering for fraud detection."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple

from ..utils import logger


class FraudFeatureEngineer:
    """Feature engineering for fraud detection."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False
        
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FraudFeatureEngineer":
        """
        Fit feature engineer on training data.
        
        Args:
            X: Features DataFrame
            y: Target (optional, not used)
            
        Returns:
            Self
        """
        logger.info("Fitting feature engineer...")
        
        # Fit scaler on Amount column if it exists
        if "Amount" in X.columns:
            self.scaler.fit(X[["Amount"]])
        
        self.fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features.
        
        Args:
            X: Features DataFrame
            
        Returns:
            Transformed features
        """
        if not self.fitted:
            raise ValueError("FeatureEngineer must be fitted before transform")
        
        X_transformed = X.copy()
        
        # Scale Amount if it exists
        if "Amount" in X_transformed.columns:
            X_transformed["Amount_Scaled"] = self.scaler.transform(X_transformed[["Amount"]])
        
        # Add time-based features if Time column exists
        if "Time" in X_transformed.columns:
            # Hour of day (assuming time is in seconds)
            X_transformed["Hour"] = (X_transformed["Time"] / 3600) % 24
            X_transformed["Is_Night"] = ((X_transformed["Hour"] >= 22) | (X_transformed["Hour"] <= 6)).astype(int)
            
        # Add amount-based features
        if "Amount" in X_transformed.columns:
            X_transformed["Amount_Log"] = np.log1p(X_transformed["Amount"])
            X_transformed["Is_Small_Transaction"] = (X_transformed["Amount"] < 10).astype(int)
            X_transformed["Is_Large_Transaction"] = (X_transformed["Amount"] > 200).astype(int)
        
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            X: Features DataFrame
            y: Target (optional, not used)
            
        Returns:
            Transformed features
        """
        return self.fit(X, y).transform(X)
