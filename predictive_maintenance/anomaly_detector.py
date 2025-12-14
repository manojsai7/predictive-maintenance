"""Anomaly detection module for predictive maintenance."""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope


class AnomalyDetector:
    """Detect anomalies in equipment sensor data."""
    
    def __init__(self, method: str = "isolation_forest", contamination: float = 0.1):
        """
        Initialize the AnomalyDetector.
        
        Args:
            method: Detection method ('isolation_forest', 'elliptic_envelope', 'statistical')
            contamination: Expected proportion of outliers in the dataset
        """
        self.method = method
        self.contamination = contamination
        self.model = None
        self.threshold_params = {}
        
        if method == "isolation_forest":
            self.model = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100
            )
        elif method == "elliptic_envelope":
            self.model = EllipticEnvelope(
                contamination=contamination,
                random_state=42
            )
    
    def fit(self, data: pd.DataFrame, feature_columns: Optional[list] = None):
        """
        Fit the anomaly detection model.
        
        Args:
            data: Training data
            feature_columns: Columns to use for detection (None for all numeric)
        """
        if feature_columns is None:
            feature_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        X = data[feature_columns].values
        
        if self.method == "statistical":
            # Statistical method using mean and std
            self.threshold_params = {
                col: {
                    'mean': data[col].mean(),
                    'std': data[col].std()
                }
                for col in feature_columns
            }
        else:
            self.model.fit(X)
        
        self.feature_columns = feature_columns
    
    def detect(self, data: pd.DataFrame) -> np.ndarray:
        """
        Detect anomalies in the data.
        
        Args:
            data: Data to check for anomalies
            
        Returns:
            Array of anomaly labels (-1 for anomaly, 1 for normal)
        """
        X = data[self.feature_columns].values
        
        if self.method == "statistical":
            # Statistical detection using 3-sigma rule
            anomalies = np.ones(len(data))
            for i, col in enumerate(self.feature_columns):
                mean = self.threshold_params[col]['mean']
                std = self.threshold_params[col]['std']
                z_scores = np.abs((X[:, i] - mean) / std) if std > 0 else np.zeros(len(X))
                anomalies[z_scores > 3] = -1
            return anomalies
        else:
            return self.model.predict(X)
    
    def detect_with_scores(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Detect anomalies and return anomaly scores.
        
        Args:
            data: Data to check for anomalies
            
        Returns:
            Dictionary with 'labels' and 'scores'
        """
        X = data[self.feature_columns].values
        labels = self.detect(data)
        
        if self.method == "statistical":
            # Calculate average z-score as anomaly score
            scores = np.zeros(len(data))
            for i, col in enumerate(self.feature_columns):
                mean = self.threshold_params[col]['mean']
                std = self.threshold_params[col]['std']
                z_scores = np.abs((X[:, i] - mean) / std) if std > 0 else np.zeros(len(X))
                scores += z_scores
            scores = scores / len(self.feature_columns)
        else:
            scores = -self.model.score_samples(X)
        
        return {
            'labels': labels,
            'scores': scores
        }
    
    def get_anomaly_indices(self, data: pd.DataFrame) -> np.ndarray:
        """
        Get indices of anomalous data points.
        
        Args:
            data: Data to check for anomalies
            
        Returns:
            Array of indices where anomalies were detected
        """
        labels = self.detect(data)
        return np.where(labels == -1)[0]
    
    def get_anomaly_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get a summary of detected anomalies.
        
        Args:
            data: Data to check for anomalies
            
        Returns:
            Dictionary containing anomaly statistics
        """
        result = self.detect_with_scores(data)
        labels = result['labels']
        scores = result['scores']
        
        anomaly_count = np.sum(labels == -1)
        anomaly_rate = anomaly_count / len(data)
        
        return {
            'total_samples': len(data),
            'anomaly_count': int(anomaly_count),
            'anomaly_rate': float(anomaly_rate),
            'mean_anomaly_score': float(np.mean(scores[labels == -1])) if anomaly_count > 0 else 0.0,
            'max_anomaly_score': float(np.max(scores)) if len(scores) > 0 else 0.0,
            'anomaly_indices': self.get_anomaly_indices(data).tolist()
        }
