"""Remaining Useful Life (RUL) prediction module."""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression


class RULPredictor:
    """Predict Remaining Useful Life (RUL) of equipment."""
    
    def __init__(self, model_type: str = "random_forest"):
        """
        Initialize the RUL Predictor.
        
        Args:
            model_type: Type of model ('random_forest', 'gradient_boosting', 'linear')
        """
        self.model_type = model_type
        self.model = None
        self.feature_columns = None
        
        if model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == "gradient_boosting":
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        elif model_type == "linear":
            self.model = LinearRegression()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def fit(
        self, 
        data: pd.DataFrame, 
        rul_column: str,
        feature_columns: Optional[list] = None
    ):
        """
        Train the RUL prediction model.
        
        Args:
            data: Training data with RUL labels
            rul_column: Name of the column containing RUL values
            feature_columns: Features to use for prediction (None for all numeric except RUL)
        """
        if feature_columns is None:
            feature_columns = [
                col for col in data.select_dtypes(include=[np.number]).columns 
                if col != rul_column
            ]
        
        self.feature_columns = feature_columns
        X = data[feature_columns].values
        y = data[rul_column].values
        
        self.model.fit(X, y)
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Predict RUL for given data.
        
        Args:
            data: Data to predict RUL for
            
        Returns:
            Array of predicted RUL values
        """
        if self.feature_columns is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        X = data[self.feature_columns].values
        predictions = self.model.predict(X)
        
        # Ensure non-negative predictions
        predictions = np.maximum(predictions, 0)
        
        return predictions
    
    def predict_with_confidence(
        self, 
        data: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        """
        Predict RUL with confidence intervals (for ensemble models).
        
        Args:
            data: Data to predict RUL for
            
        Returns:
            Dictionary with 'predictions', 'lower_bound', 'upper_bound'
        """
        predictions = self.predict(data)
        
        if self.model_type in ["random_forest", "gradient_boosting"] and hasattr(self.model, 'estimators_'):
            # Calculate prediction variance from ensemble
            X = data[self.feature_columns].values
            
            if self.model_type == "random_forest":
                tree_predictions = np.array([tree.predict(X) for tree in self.model.estimators_])
            else:
                # For gradient boosting, use staged predictions
                staged_preds = list(self.model.staged_predict(X))
                tree_predictions = np.array(staged_preds[-10:])  # Use last 10 stages
            
            std = np.std(tree_predictions, axis=0)
            
            return {
                'predictions': predictions,
                'lower_bound': np.maximum(predictions - 1.96 * std, 0),
                'upper_bound': predictions + 1.96 * std,
                'std': std
            }
        else:
            # For non-ensemble models, return predictions only
            return {
                'predictions': predictions,
                'lower_bound': predictions,
                'upper_bound': predictions,
                'std': np.zeros_like(predictions)
            }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance for tree-based models.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not hasattr(self.model, 'feature_importances_'):
            return {}
        
        importances = self.model.feature_importances_
        return {
            feature: float(importance)
            for feature, importance in zip(self.feature_columns, importances)
        }
    
    def estimate_degradation_rate(
        self, 
        data: pd.DataFrame, 
        time_column: Optional[str] = None
    ) -> float:
        """
        Estimate the average degradation rate from historical data.
        
        Args:
            data: Historical data with predictions
            time_column: Name of time column (if available)
            
        Returns:
            Estimated degradation rate (RUL decrease per time unit)
        """
        predictions = self.predict(data)
        
        if len(predictions) < 2:
            return 0.0
        
        # Calculate degradation as the change in predicted RUL over time
        rul_changes = np.diff(predictions)
        
        # Negative changes indicate degradation
        degradation_values = -rul_changes[rul_changes < 0]
        
        if len(degradation_values) == 0:
            return 0.0
        
        return float(np.mean(degradation_values))
