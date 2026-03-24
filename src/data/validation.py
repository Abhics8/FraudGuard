"""Data quality validation strictly run before training."""

import pandas as pd
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

class DataValidator:
    """Validates data schema and quality constraints before model training."""
    
    EXPECTED_COLUMNS = ["Time", "Amount", "Class"] + [f"V{i}" for i in range(1, 29)]
    
    @classmethod
    def validate_training_data(cls, df: pd.DataFrame) -> Tuple[bool, str]:
        """Run validation rules (similar to Great Expectations logic)."""
        
        # 1. Schema Validation
        missing_cols = [col for col in cls.EXPECTED_COLUMNS if col not in df.columns]
        if missing_cols:
            return False, f"Missing expected columns: {missing_cols}"
            
        # 2. Null Value Check
        null_counts = df.isnull().sum()
        if null_counts.sum() > 0:
            return False, f"Dataset contains {null_counts.sum()} null values."
            
        # 3. Value Range Validation
        if df['Amount'].min() < 0:
            return False, "Negative transaction amounts detected."
            
        if not df['Class'].isin([0, 1]).all():
            return False, "Target Class contains values other than 0 and 1."
            
        # 4. Minority Class Presence
        fraud_ratio = df['Class'].mean()
        if fraud_ratio == 0:
            return False, "No fraudulent samples found in training data. Cannot train."
            
        logger.info(f"Data validation passed! Found {len(df)} rows with {fraud_ratio:.2%} fraud ratio.")
        return True, "Success"
