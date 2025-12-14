"""Drift detection for monitoring data and model performance."""
import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from dataclasses import dataclass
from scipy import stats


@dataclass
class DriftReport:
    """Report containing drift detection results."""
    has_drift: bool
    drift_score: float
    drifted_features: List[str]
    statistics: Dict[str, float]


class DataDriftDetector:
    """Detect drift in sensor data distribution."""
    
    def __init__(self, threshold: float = 0.05):
        """
        Initialize drift detector.
        
        Args:
            threshold: P-value threshold for statistical tests
        """
        self.threshold = threshold
        self.reference_stats = {}
    
    def fit(self, reference_data: pd.DataFrame):
        """
        Fit detector on reference data.
        
        Args:
            reference_data: Reference dataset to compare against
        """
        self.reference_stats = {}
        
        for col in reference_data.select_dtypes(include=[np.number]).columns:
            self.reference_stats[col] = {
                'mean': reference_data[col].mean(),
                'std': reference_data[col].std(),
                'min': reference_data[col].min(),
                'max': reference_data[col].max(),
                'median': reference_data[col].median()
            }
    
    def detect_drift(self, current_data: pd.DataFrame) -> DriftReport:
        """
        Detect drift in current data compared to reference.
        
        Args:
            current_data: Current dataset to check for drift
            
        Returns:
            DriftReport with drift detection results
        """
        if not self.reference_stats:
            raise ValueError("Detector not fitted. Call fit() first.")
        
        drifted_features = []
        statistics = {}
        
        for col in current_data.select_dtypes(include=[np.number]).columns:
            if col not in self.reference_stats:
                continue
            
            # Use Kolmogorov-Smirnov test
            # For simplicity, compare against normal distribution with reference stats
            ref_mean = self.reference_stats[col]['mean']
            ref_std = self.reference_stats[col]['std']
            
            # Generate reference distribution samples
            reference_sample = np.random.normal(ref_mean, ref_std, len(current_data))
            current_sample = current_data[col].values
            
            # Perform KS test
            ks_statistic, p_value = stats.ks_2samp(reference_sample, current_sample)
            
            statistics[col] = {
                'ks_statistic': ks_statistic,
                'p_value': p_value
            }
            
            if p_value < self.threshold:
                drifted_features.append(col)
        
        has_drift = len(drifted_features) > 0
        drift_score = len(drifted_features) / len(self.reference_stats) if self.reference_stats else 0
        
        return DriftReport(
            has_drift=has_drift,
            drift_score=drift_score,
            drifted_features=drifted_features,
            statistics=statistics
        )
    
    def detect_concept_drift(self, y_true: np.ndarray, y_pred: np.ndarray,
                           window_size: int = 100) -> bool:
        """
        Detect concept drift by monitoring prediction errors.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            window_size: Size of sliding window
            
        Returns:
            True if concept drift detected, False otherwise
        """
        errors = np.abs(y_true - y_pred)
        
        if len(errors) < window_size * 2:
            return False
        
        # Compare recent window with older window
        recent_errors = errors[-window_size:]
        older_errors = errors[-2*window_size:-window_size]
        
        # Use Mann-Whitney U test
        _, p_value = stats.mannwhitneyu(recent_errors, older_errors)
        
        return p_value < self.threshold


class ModelPerformanceMonitor:
    """Monitor model performance metrics over time."""
    
    def __init__(self):
        """Initialize performance monitor."""
        self.metrics_history = []
    
    def log_metrics(self, metrics: Dict[str, float], timestamp: Optional[str] = None):
        """
        Log performance metrics.
        
        Args:
            metrics: Dictionary of metric names and values
            timestamp: Optional timestamp for the metrics
        """
        if timestamp is None:
            timestamp = pd.Timestamp.now().isoformat()
        
        metrics_entry = {'timestamp': timestamp, **metrics}
        self.metrics_history.append(metrics_entry)
    
    def get_metrics_dataframe(self) -> pd.DataFrame:
        """
        Get metrics history as DataFrame.
        
        Returns:
            DataFrame with metrics over time
        """
        return pd.DataFrame(self.metrics_history)
    
    def detect_performance_degradation(self, metric_name: str,
                                      threshold: float = 0.1) -> bool:
        """
        Detect if model performance has degraded.
        
        Args:
            metric_name: Name of metric to monitor
            threshold: Degradation threshold (percentage)
            
        Returns:
            True if performance degradation detected
        """
        if len(self.metrics_history) < 2:
            return False
        
        df = self.get_metrics_dataframe()
        
        if metric_name not in df.columns:
            return False
        
        recent_avg = df[metric_name].iloc[-10:].mean()
        baseline_avg = df[metric_name].iloc[:10].mean()
        
        # Check if performance has degraded by more than threshold
        degradation = (baseline_avg - recent_avg) / baseline_avg
        
        return degradation > threshold
    
    def get_summary_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics for all metrics.
        
        Returns:
            Dictionary with statistics for each metric
        """
        df = self.get_metrics_dataframe()
        
        summary = {}
        for col in df.columns:
            if col != 'timestamp' and pd.api.types.is_numeric_dtype(df[col]):
                summary[col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'current': df[col].iloc[-1] if len(df) > 0 else None
                }
        
        return summary


class AlertManager:
    """Manage alerts for drift and performance issues."""
    
    def __init__(self):
        """Initialize alert manager."""
        self.alerts = []
    
    def create_alert(self, alert_type: str, severity: str, message: str,
                    metadata: Optional[Dict] = None):
        """
        Create a new alert.
        
        Args:
            alert_type: Type of alert (e.g., 'drift', 'performance')
            severity: Severity level ('low', 'medium', 'high', 'critical')
            message: Alert message
            metadata: Optional metadata dictionary
        """
        alert = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'type': alert_type,
            'severity': severity,
            'message': message,
            'metadata': metadata or {}
        }
        self.alerts.append(alert)
    
    def get_alerts(self, severity: Optional[str] = None,
                  alert_type: Optional[str] = None) -> List[Dict]:
        """
        Get filtered alerts.
        
        Args:
            severity: Filter by severity level
            alert_type: Filter by alert type
            
        Returns:
            List of alerts matching filters
        """
        filtered = self.alerts
        
        if severity:
            filtered = [a for a in filtered if a['severity'] == severity]
        
        if alert_type:
            filtered = [a for a in filtered if a['type'] == alert_type]
        
        return filtered
    
    def clear_alerts(self):
        """Clear all alerts."""
        self.alerts = []
    
    def get_alert_summary(self) -> Dict[str, int]:
        """
        Get summary of alerts by severity.
        
        Returns:
            Dictionary with counts by severity level
        """
        summary = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        
        for alert in self.alerts:
            severity = alert['severity']
            if severity in summary:
                summary[severity] += 1
        
        return summary
