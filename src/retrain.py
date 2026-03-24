"""Automated retraining pipeline triggered when drift is detected."""

import sys
import logging
from pathlib import Path
import subprocess

logger = logging.getLogger(__name__)

def check_drift_and_retrain(drift_score: float, threshold: float = 0.15):
    """
    Check if data/concept drift exceeds threshold.
    If so, kick off automated retraining pipeline.
    """
    logger.info(f"Checking drift score: {drift_score:.3f} against threshold: {threshold}")
    
    if drift_score > threshold:
        logger.warning(f"🚨 ALERT: Drift threshold exceeded ({drift_score:.3f} > {threshold})! Triggering retraining.")
        
        # In a real environment, this might trigger an Airflow DAG or GitHub Actions workflow webhook
        # Here we simulate triggering train.py
        try:
            logger.info("Executing training pipeline...")
            result = subprocess.run(
                [sys.executable, "train.py"], 
                capture_output=True, 
                text=True, 
                check=True
            )
            logger.info("Retraining successful!")
            logger.debug(result.stdout)
            
            # After successful training, we would promote the model in the registry
            logger.info("New model registered in MLflow 'staging' environment.")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Retraining failed: {e.stderr}")
            sys.exit(1)
    else:
        logger.info("Drift within normal limits. No retraining required.")

if __name__ == "__main__":
    # Simulate a drift alert payload
    simulated_drift_input = 0.22  # > 0.15
    check_drift_and_retrain(simulated_drift_input)
