"""Model registry and A/B testing router for FraudGuard."""

import random
import logging
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

class ModelRegistry:
    """Manages model promotion stages and MLflow integration."""
    
    def __init__(self):
        self.models = {
            "production": None,  # Primary model (e.g., XGBoost)
            "staging": None,     # Challenger model (e.g., LightGBM)
            "dev": None          # Experimental model
        }
        
    def promote_model(self, model_version: str, stage: str):
        """Promote a model version to a specific stage in MLflow."""
        logger.info(f"Promoting model version {model_version} to {stage} stage.")
        # In actual MLflow usage:
        # client = MlflowClient()
        # client.transition_model_version_stage(name="FraudGuard", version=model_version, stage=stage)
        self.models[stage] = f"model_v{model_version}"
        return True

    def load_stage_model(self, stage: str = "production"):
        """Load the active model for a specific stage."""
        # Simulated model load
        if stage == "production":
            return {"name": "XGBoost_Prod", "predict_proba": lambda x: [0.15] * len(x)}
        elif stage == "staging":
            return {"name": "LightGBM_Staging", "predict_proba": lambda x: [0.12] * len(x)}
        return None


class ABTestRouter:
    """Routes traffic between Production (Champion) and Staging (Challenger) models."""
    
    def __init__(self, staging_traffic_pct: int = 10):
        self.staging_traffic_pct = staging_traffic_pct
        self.metrics = {"production": {"requests": 0, "fraud_detected": 0},
                        "staging": {"requests": 0, "fraud_detected": 0}}
        
    def route_request(self) -> str:
        """Decide whether to route to production or staging based on traffic split."""
        if random.randint(1, 100) <= self.staging_traffic_pct:
            return "staging"
        return "production"
        
    def log_prediction(self, stage: str, is_fraud: bool):
        """Log prediction outcomes for A/B test analysis."""
        self.metrics[stage]["requests"] += 1
        if is_fraud:
            self.metrics[stage]["fraud_detected"] += 1
            
    def get_comparison_metrics(self) -> Dict[str, Any]:
        """Return precision/recall insights collected during shadow mode or A/B testing."""
        return self.metrics
