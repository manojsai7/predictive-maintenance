"""Failure prediction module for predictive maintenance."""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


class FailurePredictor:
    """Predict equipment failures before they occur."""
    
    def __init__(self, model_type: str = "random_forest", prediction_horizon: int = 10):
        """
        Initialize the Failure Predictor.
        
        Args:
            model_type: Type of model ('random_forest', 'gradient_boosting', 'logistic')
            prediction_horizon: Time steps ahead to predict failures
        """
        self.model_type = model_type
        self.prediction_horizon = prediction_horizon
        self.model = None
        self.feature_columns = None
        
        if model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
        elif model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        elif model_type == "logistic":
            self.model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced'
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def fit(
        self, 
        data: pd.DataFrame, 
        failure_column: str,
        feature_columns: Optional[list] = None
    ):
        """
        Train the failure prediction model.
        
        Args:
            data: Training data with failure labels
            failure_column: Name of the column indicating failures (0=normal, 1=failure)
            feature_columns: Features to use for prediction (None for all numeric except failure)
        """
        if feature_columns is None:
            feature_columns = [
                col for col in data.select_dtypes(include=[np.number]).columns 
                if col != failure_column
            ]
        
        self.feature_columns = feature_columns
        X = data[feature_columns].values
        y = data[failure_column].values
        
        self.model.fit(X, y)
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Predict failure occurrence.
        
        Args:
            data: Data to predict failures for
            
        Returns:
            Array of binary predictions (0=normal, 1=failure)
        """
        if self.feature_columns is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        X = data[self.feature_columns].values
        return self.model.predict(X)
    
    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """
        Predict failure probability.
        
        Args:
            data: Data to predict failure probability for
            
        Returns:
            Array of failure probabilities (probability of class 1)
        """
        if self.feature_columns is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        X = data[self.feature_columns].values
        proba = self.model.predict_proba(X)
        
        # Return probability of failure (class 1)
        return proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
    
    def predict_with_confidence(
        self, 
        data: pd.DataFrame,
        threshold: float = 0.5
    ) -> Dict[str, np.ndarray]:
        """
        Predict failures with probability and confidence scores.
        
        Args:
            data: Data to predict failures for
            threshold: Probability threshold for classifying as failure
            
        Returns:
            Dictionary with 'predictions', 'probabilities', 'risk_level'
        """
        probabilities = self.predict_proba(data)
        predictions = (probabilities >= threshold).astype(int)
        
        # Categorize risk levels
        risk_levels = np.zeros(len(probabilities), dtype=int)
        risk_levels[probabilities < 0.3] = 0  # Low risk
        risk_levels[(probabilities >= 0.3) & (probabilities < 0.6)] = 1  # Medium risk
        risk_levels[(probabilities >= 0.6) & (probabilities < 0.8)] = 2  # High risk
        risk_levels[probabilities >= 0.8] = 3  # Critical risk
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'risk_level': risk_levels
        }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance for tree-based models.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not hasattr(self.model, 'feature_importances_'):
            if hasattr(self.model, 'coef_'):
                # For logistic regression, use absolute coefficients
                importances = np.abs(self.model.coef_[0])
                return {
                    feature: float(importance)
                    for feature, importance in zip(self.feature_columns, importances)
                }
            return {}
        
        importances = self.model.feature_importances_
        return {
            feature: float(importance)
            for feature, importance in zip(self.feature_columns, importances)
        }
    
    def get_failure_risk_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get a comprehensive summary of failure risks.
        
        Args:
            data: Data to analyze
            
        Returns:
            Dictionary containing risk statistics
        """
        result = self.predict_with_confidence(data)
        probabilities = result['probabilities']
        risk_levels = result['risk_level']
        
        return {
            'total_samples': len(data),
            'predicted_failures': int(np.sum(result['predictions'])),
            'failure_rate': float(np.mean(result['predictions'])),
            'average_failure_probability': float(np.mean(probabilities)),
            'max_failure_probability': float(np.max(probabilities)),
            'low_risk_count': int(np.sum(risk_levels == 0)),
            'medium_risk_count': int(np.sum(risk_levels == 1)),
            'high_risk_count': int(np.sum(risk_levels == 2)),
            'critical_risk_count': int(np.sum(risk_levels == 3)),
            'high_risk_indices': np.where(risk_levels >= 2)[0].tolist()
        }
    
    def estimate_time_to_failure(
        self, 
        current_probability: float,
        degradation_rate: float = 0.01
    ) -> float:
        """
        Estimate time until failure based on current probability and degradation rate.
        
        Args:
            current_probability: Current failure probability
            degradation_rate: Rate of probability increase per time unit
            
        Returns:
            Estimated time steps until failure (probability >= 0.9)
        """
        if current_probability >= 0.9:
            return 0.0
        
        if degradation_rate <= 0:
            return float('inf')
        
        # Simple linear estimation
        time_to_failure = (0.9 - current_probability) / degradation_rate
        return max(0.0, time_to_failure)
