"""Drift detection using Evidently AI."""

import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
from datetime import datetime
import json

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.metrics import (
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
    ColumnDriftMetric
)

from ..utils import logger, settings


class DriftDetector:
    """Monitor data and concept drift using Evidently AI."""
    
    def __init__(self, drift_threshold: float = 0.05):
        """
        Initialize drift detector.
        
        Args:
            drift_threshold: Threshold for drift detection (default: 0.05)
        """
        self.drift_threshold = drift_threshold
        self.reference_data: Optional[pd.DataFrame] = None
        self.reports_dir = Path("reports/drift")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
    def set_reference(self, reference_data: pd.DataFrame):
        """
        Set reference dataset (typically training data).
        
        Args:
            reference_data: Reference DataFrame
        """
        self.reference_data = reference_data
        logger.info(f"Reference data set: {len(reference_data)} samples")
        
    def detect_data_drift(
        self,
        current_data: pd.DataFrame,
        save_report: bool = True
    ) -> Dict[str, any]:
        """
        Detect data drift between reference and current data.
        
        Args:
            current_data: Current production data
            save_report: Whether to save HTML report
            
        Returns:
            Dictionary with drift metrics
        """
        if self.reference_data is None:
            raise ValueError("Reference data not set. Call set_reference() first.")
        
        logger.info("Detecting data drift...")
        
        # Create drift report
        report = Report(metrics=[
            DataDriftPreset(),
            DatasetDriftMetric(),
            DatasetMissingValuesMetric()
        ])
        
        # Run report
        report.run(
            reference_data=self.reference_data,
            current_data=current_data
        )
        
        # Get results
        results = report.as_dict()
        
        # Extract key metrics
        drift_detected = results['metrics'][1]['result']['dataset_drift']
        drift_share = results['metrics'][1]['result']['drift_share']
        
        drift_summary = {
            'drift_detected': drift_detected,
            'drift_share': drift_share,
            'timestamp': datetime.now().isoformat(),
            'reference_size': len(self.reference_data),
            'current_size': len(current_data),
            'threshold': self.drift_threshold
        }
        
        # Save report if requested
        if save_report:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = self.reports_dir / f"data_drift_{timestamp}.html"
            report.save_html(str(report_path))
            logger.info(f"Drift report saved to {report_path}")
            drift_summary['report_path'] = str(report_path)
        
        # Log results
        if drift_detected:
            logger.warning(f"⚠️ Data drift detected! Drift share: {drift_share:.2%}")
        else:
            logger.info(f"✅ No data drift detected. Drift share: {drift_share:.2%}")
        
        return drift_summary
    
    def detect_target_drift(
        self,
        current_data: pd.DataFrame,
        target_column: str = "Class",
        save_report: bool = True
    ) -> Dict[str, any]:
        """
        Detect target (prediction) drift.
        
        Args:
            current_data: Current production data with predictions
            target_column: Name of target column
            save_report: Whether to save HTML report
            
        Returns:
            Dictionary with target drift metrics
        """
        if self.reference_data is None:
            raise ValueError("Reference data not set. Call set_reference() first.")
        
        logger.info("Detecting target drift...")
        
        # Create target drift report
        report = Report(metrics=[
            TargetDriftPreset()
        ])
        
        # Run report
        report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping={'target': target_column}
        )
        
        # Get results
        results = report.as_dict()
        
        # Extract metrics
        target_drift_detected = results['metrics'][0]['result']['drift_detected']
        
        drift_summary = {
            'target_drift_detected': target_drift_detected,
            'timestamp': datetime.now().isoformat(),
            'target_column': target_column
        }
        
        # Save report
        if save_report:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = self.reports_dir / f"target_drift_{timestamp}.html"
            report.save_html(str(report_path))
            logger.info(f"Target drift report saved to {report_path}")
            drift_summary['report_path'] = str(report_path)
        
        # Log results
        if target_drift_detected:
            logger.warning(f"⚠️ Target drift detected!")
        else:
            logger.info(f"✅ No target drift detected")
        
        return drift_summary
    
    def monitor_column_drift(
        self,
        current_data: pd.DataFrame,
        columns: list[str],
        save_report: bool = True
    ) -> Dict[str, any]:
        """
        Monitor drift for specific columns.
        
        Args:
            current_data: Current production data
            columns: List of column names to monitor
            save_report: Whether to save HTML report
            
        Returns:
            Dictionary with column-wise drift metrics
        """
        if self.reference_data is None:
            raise ValueError("Reference data not set. Call set_reference() first.")
        
        logger.info(f"Monitoring drift for {len(columns)} columns...")
        
        # Create column drift metrics
        metrics = [
            ColumnDriftMetric(column_name=col, stattest='ks')
            for col in columns
        ]
        
        report = Report(metrics=metrics)
        
        # Run report
        report.run(
            reference_data=self.reference_data,
            current_data=current_data
        )
        
        # Get results
        results = report.as_dict()
        
        # Extract drift per column
        column_drifts = {}
        drifted_columns = []
        
        for i, col in enumerate(columns):
            metric = results['metrics'][i]['result']
            drift_detected = metric['drift_detected']
            drift_score = metric.get('drift_score', 0)
            
            column_drifts[col] = {
                'drift_detected': drift_detected,
                'drift_score': drift_score
            }
            
            if drift_detected:
                drifted_columns.append(col)
        
        summary = {
            'columns_checked': len(columns),
            'columns_drifted': len(drifted_columns),
            'drifted_columns': drifted_columns,
            'column_drifts': column_drifts,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save report
        if save_report:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = self.reports_dir / f"column_drift_{timestamp}.html"
            report.save_html(str(report_path))
            summary['report_path'] = str(report_path)
        
        # Log results
        if drifted_columns:
            logger.warning(f"⚠️ Drift detected in {len(drifted_columns)} columns: {drifted_columns}")
        else:
            logger.info(f"✅ No column drift detected")
        
        return summary
    
    def should_retrain(
        self,
        current_data: pd.DataFrame,
        columns_to_check: list[str] | None = None
    ) -> Tuple[bool, str]:
        """
        Determine if model should be retrained based on drift.
        
        Args:
            current_data: Current production data
            columns_to_check: Specific columns to monitor (optional)
            
        Returns:
            Tuple of (should_retrain: bool, reason: str)
        """
        # Check data drift
        data_drift = self.detect_data_drift(current_data, save_report=True)
        
        if data_drift['drift_detected']:
            return True, f"Data drift detected (drift_share: {data_drift['drift_share']:.2%})"
        
        # Check column drift if specified
        if columns_to_check:
            column_drift = self.monitor_column_drift(
                current_data,
                columns_to_check,
                save_report=True
            )
            
            if column_drift['columns_drifted'] > 0:
                return True, f"Drift in {column_drift['columns_drifted']} columns: {column_drift['drifted_columns']}"
        
        return False, "No significant drift detected"
