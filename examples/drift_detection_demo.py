"""Example script demonstrating drift detection."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data import load_credit_card_fraud_data, split_data
from src.monitoring import DriftDetector
from src.utils import logger


def simulate_drift(df: pd.DataFrame, drift_strength: float = 0.5) -> pd.DataFrame:
    """
    Simulate data drift by modifying feature distributions.
    
    Args:
        df: Original DataFrame
        drift_strength: Strength of drift (0-1)
        
    Returns:
        DataFrame with simulated drift
    """
    drifted_df = df.copy()
    
    # Add drift to Amount column
    drifted_df['Amount'] = drifted_df['Amount'] * (1 + np.random.normal(drift_strength, 0.1, len(df)))
    
    # Add drift to some V columns
    for col in ['V1', 'V2', 'V3']:
        drifted_df[col] = drifted_df[col] + np.random.normal(drift_strength, 0.2, len(df))
    
    return drifted_df


def main():
    """Demonstrate drift detection."""
    logger.info("Loading data...")
    
    try:
        df = load_credit_card_fraud_data()
    except FileNotFoundError:
        logger.error("Dataset not found! Please download it first.")
        return
    
    # Split data
    train_df, val_df, test_df = split_data(df, test_size=0.2, val_size=0.1)
    
    # Initialize drift detector
    logger.info("\n" + "="*50)
    logger.info("Initializing Drift Detector")
    logger.info("="*50)
    
    detector = DriftDetector(drift_threshold=0.05)
    detector.set_reference(train_df)
    
    # Test 1: No drift (validation data from same distribution)
    logger.info("\n" + "="*50)
    logger.info("Test 1: Detecting drift on validation data (no drift expected)")
    logger.info("="*50)
    
    drift_result = detector.detect_data_drift(val_df, save_report=True)
    logger.info(f"\nDrift detected: {drift_result['drift_detected']}")
    logger.info(f"Drift share: {drift_result['drift_share']:.2%}")
    
    # Test 2: Simulated drift
    logger.info("\n" + "="*50)
    logger.info("Test 2: Detecting drift on simulated drifted data")
    logger.info("="*50)
    
    drifted_data = simulate_drift(val_df, drift_strength=0.5)
    drift_result = detector.detect_data_drift(drifted_data, save_report=True)
    logger.info(f"\nDrift detected: {drift_result['drift_detected']}")
    logger.info(f"Drift share: {drift_result['drift_share']:.2%}")
    
    # Test 3: Column-level drift monitoring
    logger.info("\n" + "="*50)
    logger.info("Test 3: Monitoring column-level drift")
    logger.info("="*50)
    
    columns_to_monitor = ['Amount', 'V1', 'V2', 'V3', 'V4', 'V5']
    column_drift = detector.monitor_column_drift(
        drifted_data,
        columns_to_monitor,
        save_report=True
    )
    
    logger.info(f"\nColumns checked: {column_drift['columns_checked']}")
    logger.info(f"Columns with drift: {column_drift['columns_drifted']}")
    if column_drift['drifted_columns']:
        logger.info(f"Drifted columns: {column_drift['drifted_columns']}")
    
    # Test 4: Should retrain decision
    logger.info("\n" + "="*50)
    logger.info("Test 4: Automated retrain decision")
    logger.info("="*50)
    
    should_retrain, reason = detector.should_retrain(drifted_data, columns_to_monitor)
    
    if should_retrain:
        logger.warning(f"🔄 Model should be retrained: {reason}")
    else:
        logger.info(f"✅ Model is still valid: {reason}")
    
    logger.info("\n" + "="*50)
    logger.info("Drift Detection Complete!")
    logger.info(f"Reports saved to: {detector.reports_dir}")
    logger.info("="*50)


if __name__ == "__main__":
    main()
