"""Classical machine learning models for predictive maintenance."""
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib


class RULPredictor:
    """Predict Remaining Useful Life using classical ML models."""
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize RUL predictor.
        
        Args:
            model_type: Type of model to use ('random_forest' or 'xgboost')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def build_model(self, **model_params):
        """
        Build the prediction model.
        
        Args:
            **model_params: Parameters to pass to the model
        """
        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=model_params.get('n_estimators', 100),
                max_depth=model_params.get('max_depth', 10),
                random_state=model_params.get('random_state', 42),
                n_jobs=-1
            )
        elif self.model_type == 'xgboost':
            self.model = xgb.XGBRegressor(
                n_estimators=model_params.get('n_estimators', 100),
                max_depth=model_params.get('max_depth', 6),
                learning_rate=model_params.get('learning_rate', 0.1),
                random_state=model_params.get('random_state', 42)
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X: pd.DataFrame, y: np.ndarray):
        """
        Train the RUL prediction model.
        
        Args:
            X: Training features
            y: Training labels (RUL values)
        """
        if self.model is None:
            self.build_model()
        
        self.feature_names = X.columns.tolist()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict RUL for new data.
        
        Args:
            X: Input features
            
        Returns:
            Predicted RUL values
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Ensure same features
        if self.feature_names:
            X = X[self.feature_names]
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        # Ensure non-negative predictions
        predictions = np.maximum(predictions, 0)
        
        return predictions
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained.")
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            return dict(zip(self.feature_names, importance))
        else:
            return {}
    
    def save_model(self, path: str):
        """
        Save model to disk.
        
        Args:
            path: Path to save the model
        """
        model_dict = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type
        }
        joblib.dump(model_dict, path)
    
    def load_model(self, path: str):
        """
        Load model from disk.
        
        Args:
            path: Path to load the model from
        """
        model_dict = joblib.load(path)
        self.model = model_dict['model']
        self.scaler = model_dict['scaler']
        self.feature_names = model_dict['feature_names']
        self.model_type = model_dict['model_type']


class FailureClassifier:
    """Binary classification for failure prediction."""
    
    def __init__(self, model_type: str = 'random_forest', threshold: float = 0.5):
        """
        Initialize failure classifier.
        
        Args:
            model_type: Type of model to use ('random_forest' or 'xgboost')
            threshold: Classification threshold
        """
        self.model_type = model_type
        self.threshold = threshold
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
    
    def build_model(self, **model_params):
        """Build the classification model."""
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=model_params.get('n_estimators', 100),
                max_depth=model_params.get('max_depth', 10),
                random_state=model_params.get('random_state', 42),
                n_jobs=-1
            )
        elif self.model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators=model_params.get('n_estimators', 100),
                max_depth=model_params.get('max_depth', 6),
                learning_rate=model_params.get('learning_rate', 0.1),
                random_state=model_params.get('random_state', 42)
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X: pd.DataFrame, y: np.ndarray):
        """
        Train the failure classification model.
        
        Args:
            X: Training features
            y: Training labels (0=normal, 1=failure)
        """
        if self.model is None:
            self.build_model()
        
        self.feature_names = X.columns.tolist()
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict failure probability.
        
        Args:
            X: Input features
            
        Returns:
            Binary predictions (0=normal, 1=failure)
        """
        if self.model is None:
            raise ValueError("Model not trained.")
        
        if self.feature_names:
            X = X[self.feature_names]
        
        X_scaled = self.scaler.transform(X)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
        return (probabilities >= self.threshold).astype(int)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict failure probability.
        
        Args:
            X: Input features
            
        Returns:
            Probability of failure
        """
        if self.model is None:
            raise ValueError("Model not trained.")
        
        if self.feature_names:
            X = X[self.feature_names]
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def save_model(self, path: str):
        """Save model to disk."""
        model_dict = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'threshold': self.threshold
        }
        joblib.dump(model_dict, path)
    
    def load_model(self, path: str):
        """Load model from disk."""
        model_dict = joblib.load(path)
        self.model = model_dict['model']
        self.scaler = model_dict['scaler']
        self.feature_names = model_dict['feature_names']
        self.model_type = model_dict['model_type']
        self.threshold = model_dict['threshold']
