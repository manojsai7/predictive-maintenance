"""Health scoring module for equipment condition assessment."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List


class HealthScorer:
    """Calculate and track equipment health scores."""
    
    def __init__(self):
        """Initialize the HealthScorer."""
        self.baseline_values = {}
        self.weights = {}
        self.thresholds = {
            'excellent': 90,
            'good': 75,
            'fair': 60,
            'poor': 40,
            'critical': 0
        }
    
    def set_baseline(self, data: pd.DataFrame, columns: List[str]):
        """
        Set baseline values for normal operation.
        
        Args:
            data: Historical data representing normal operation
            columns: Sensor/metric columns to track
        """
        for col in columns:
            self.baseline_values[col] = {
                'mean': data[col].mean(),
                'std': data[col].std(),
                'min': data[col].min(),
                'max': data[col].max()
            }
    
    def set_weights(self, weights: Dict[str, float]):
        """
        Set importance weights for different metrics.
        
        Args:
            weights: Dictionary mapping metric names to weights (should sum to 1.0)
        """
        total = sum(weights.values())
        self.weights = {k: v/total for k, v in weights.items()}
    
    def calculate_metric_health(
        self, 
        value: float, 
        metric_name: str
    ) -> float:
        """
        Calculate health score for a single metric.
        
        Args:
            value: Current value of the metric
            metric_name: Name of the metric
            
        Returns:
            Health score between 0 and 100
        """
        if metric_name not in self.baseline_values:
            return 100.0  # Default to perfect health if no baseline
        
        baseline = self.baseline_values[metric_name]
        mean = baseline['mean']
        std = baseline['std']
        
        if std == 0:
            return 100.0 if value == mean else 50.0
        
        # Calculate deviation from baseline (in standard deviations)
        z_score = abs((value - mean) / std)
        
        # Convert z-score to health score (inverse relationship)
        # 0 std dev = 100 health, 3+ std dev = 0 health
        health = max(0, 100 - (z_score / 3) * 100)
        
        return health
    
    def calculate_overall_health(
        self, 
        data: pd.Series,
        metric_names: Optional[List[str]] = None
    ) -> float:
        """
        Calculate overall health score from multiple metrics.
        
        Args:
            data: Series containing current metric values
            metric_names: List of metrics to include (None for all)
            
        Returns:
            Overall health score between 0 and 100
        """
        if metric_names is None:
            metric_names = list(self.baseline_values.keys())
        
        health_scores = []
        weights = []
        
        for metric in metric_names:
            if metric in data.index:
                health = self.calculate_metric_health(data[metric], metric)
                weight = self.weights.get(metric, 1.0)
                health_scores.append(health)
                weights.append(weight)
        
        if not health_scores:
            return 100.0
        
        # Weighted average
        total_weight = sum(weights)
        overall_health = sum(h * w for h, w in zip(health_scores, weights)) / total_weight
        
        return overall_health
    
    def calculate_health_batch(
        self, 
        data: pd.DataFrame,
        metric_names: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Calculate health scores for multiple data points.
        
        Args:
            data: DataFrame containing metric values
            metric_names: List of metrics to include (None for all)
            
        Returns:
            Array of health scores
        """
        health_scores = []
        for _, row in data.iterrows():
            health = self.calculate_overall_health(row, metric_names)
            health_scores.append(health)
        
        return np.array(health_scores)
    
    def get_health_category(self, health_score: float) -> str:
        """
        Get categorical health status from numeric score.
        
        Args:
            health_score: Numeric health score (0-100)
            
        Returns:
            Health category string
        """
        if health_score >= self.thresholds['excellent']:
            return 'excellent'
        elif health_score >= self.thresholds['good']:
            return 'good'
        elif health_score >= self.thresholds['fair']:
            return 'fair'
        elif health_score >= self.thresholds['poor']:
            return 'poor'
        else:
            return 'critical'
    
    def get_health_trend(
        self, 
        health_scores: np.ndarray,
        window: int = 10
    ) -> str:
        """
        Determine health trend over time.
        
        Args:
            health_scores: Array of historical health scores
            window: Number of recent scores to consider
            
        Returns:
            Trend indicator ('improving', 'stable', 'degrading')
        """
        if len(health_scores) < 2:
            return 'stable'
        
        recent_scores = health_scores[-window:]
        
        if len(recent_scores) < 2:
            return 'stable'
        
        # Calculate linear trend
        x = np.arange(len(recent_scores))
        slope = np.polyfit(x, recent_scores, 1)[0]
        
        # Classify trend based on slope
        if slope > 1.0:
            return 'improving'
        elif slope < -1.0:
            return 'degrading'
        else:
            return 'stable'
    
    def get_health_report(
        self, 
        data: pd.Series,
        metric_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive health report.
        
        Args:
            data: Current metric values
            metric_names: List of metrics to include
            
        Returns:
            Dictionary containing health report
        """
        overall_health = self.calculate_overall_health(data, metric_names)
        category = self.get_health_category(overall_health)
        
        metric_scores = {}
        if metric_names is None:
            metric_names = list(self.baseline_values.keys())
        
        for metric in metric_names:
            if metric in data.index:
                metric_scores[metric] = self.calculate_metric_health(data[metric], metric)
        
        return {
            'overall_health_score': float(overall_health),
            'health_category': category,
            'metric_scores': {k: float(v) for k, v in metric_scores.items()},
            'timestamp': pd.Timestamp.now().isoformat()
        }
