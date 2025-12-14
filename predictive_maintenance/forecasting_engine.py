"""Forecasting engine for predictive maintenance."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from sklearn.ensemble import RandomForestRegressor


class ForecastingEngine:
    """Forecast equipment metrics and future states."""
    
    def __init__(self, forecast_horizon: int = 10):
        """
        Initialize the ForecastingEngine.
        
        Args:
            forecast_horizon: Number of time steps to forecast ahead
        """
        self.forecast_horizon = forecast_horizon
        self.models = {}
        self.feature_columns = []
    
    def prepare_sequences(
        self, 
        data: pd.DataFrame,
        target_column: str,
        lookback: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare time series sequences for forecasting.
        
        Args:
            data: Time series data
            target_column: Column to forecast
            lookback: Number of past time steps to use as features
            
        Returns:
            Tuple of (X, y) arrays
        """
        values = data[target_column].values
        X, y = [], []
        
        for i in range(lookback, len(values)):
            X.append(values[i-lookback:i])
            y.append(values[i])
        
        return np.array(X), np.array(y)
    
    def fit(
        self, 
        data: pd.DataFrame,
        target_columns: List[str],
        lookback: int = 10
    ):
        """
        Train forecasting models for specified columns.
        
        Args:
            data: Historical time series data
            target_columns: Columns to create forecasting models for
            lookback: Number of past time steps to use as features
        """
        self.lookback = lookback
        
        for col in target_columns:
            X, y = self.prepare_sequences(data, col, lookback)
            
            if len(X) == 0:
                continue
            
            model = RandomForestRegressor(
                n_estimators=50,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X, y)
            self.models[col] = model
    
    def forecast_single_step(
        self, 
        history: np.ndarray,
        column: str
    ) -> float:
        """
        Forecast one step ahead for a single metric.
        
        Args:
            history: Array of recent values (length = lookback)
            column: Name of the column to forecast
            
        Returns:
            Forecasted value
        """
        if column not in self.models:
            raise ValueError(f"No model trained for column: {column}")
        
        model = self.models[column]
        X = history[-self.lookback:].reshape(1, -1)
        
        return model.predict(X)[0]
    
    def forecast_multi_step(
        self, 
        history: np.ndarray,
        column: str,
        steps: Optional[int] = None
    ) -> np.ndarray:
        """
        Forecast multiple steps ahead for a single metric.
        
        Args:
            history: Array of recent values (length >= lookback)
            column: Name of the column to forecast
            steps: Number of steps to forecast (None for forecast_horizon)
            
        Returns:
            Array of forecasted values
        """
        if steps is None:
            steps = self.forecast_horizon
        
        forecasts = []
        current_history = history.copy()
        
        for _ in range(steps):
            next_value = self.forecast_single_step(current_history, column)
            forecasts.append(next_value)
            current_history = np.append(current_history, next_value)
        
        return np.array(forecasts)
    
    def forecast_dataframe(
        self, 
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        steps: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Forecast multiple metrics and return as DataFrame.
        
        Args:
            data: Historical data containing all columns
            columns: Columns to forecast (None for all trained columns)
            steps: Number of steps to forecast
            
        Returns:
            DataFrame with forecasted values
        """
        if columns is None:
            columns = list(self.models.keys())
        
        if steps is None:
            steps = self.forecast_horizon
        
        forecasts = {}
        
        for col in columns:
            if col not in self.models:
                continue
            
            history = data[col].values
            forecast = self.forecast_multi_step(history, col, steps)
            forecasts[col] = forecast
        
        return pd.DataFrame(forecasts)
    
    def forecast_with_confidence(
        self, 
        data: pd.DataFrame,
        column: str,
        steps: Optional[int] = None,
        n_iterations: int = 100
    ) -> Dict[str, np.ndarray]:
        """
        Forecast with confidence intervals using bootstrap.
        
        Args:
            data: Historical data
            column: Column to forecast
            steps: Number of steps to forecast
            n_iterations: Number of bootstrap iterations
            
        Returns:
            Dictionary with 'forecast', 'lower_bound', 'upper_bound'
        """
        if steps is None:
            steps = self.forecast_horizon
        
        history = data[column].values
        forecasts_all = []
        
        # Generate multiple forecasts with noise
        for _ in range(n_iterations):
            # Add small noise to history
            noisy_history = history + np.random.normal(0, history.std() * 0.05, len(history))
            forecast = self.forecast_multi_step(noisy_history, column, steps)
            forecasts_all.append(forecast)
        
        forecasts_all = np.array(forecasts_all)
        
        return {
            'forecast': np.mean(forecasts_all, axis=0),
            'lower_bound': np.percentile(forecasts_all, 2.5, axis=0),
            'upper_bound': np.percentile(forecasts_all, 97.5, axis=0),
            'std': np.std(forecasts_all, axis=0)
        }
    
    def detect_forecast_anomalies(
        self, 
        forecast: np.ndarray,
        thresholds: Dict[str, float]
    ) -> List[int]:
        """
        Detect time steps where forecasted values exceed thresholds.
        
        Args:
            forecast: Array of forecasted values
            thresholds: Dictionary with 'min' and 'max' thresholds
            
        Returns:
            List of time step indices with anomalies
        """
        anomaly_indices = []
        
        for i, value in enumerate(forecast):
            if 'min' in thresholds and value < thresholds['min']:
                anomaly_indices.append(i)
            elif 'max' in thresholds and value > thresholds['max']:
                anomaly_indices.append(i)
        
        return anomaly_indices
    
    def get_forecast_summary(
        self, 
        data: pd.DataFrame,
        column: str,
        steps: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive forecast summary.
        
        Args:
            data: Historical data
            column: Column to forecast
            steps: Number of steps to forecast
            
        Returns:
            Dictionary containing forecast information
        """
        forecast_result = self.forecast_with_confidence(data, column, steps)
        
        return {
            'column': column,
            'forecast_horizon': steps or self.forecast_horizon,
            'forecast_values': forecast_result['forecast'].tolist(),
            'confidence_lower': forecast_result['lower_bound'].tolist(),
            'confidence_upper': forecast_result['upper_bound'].tolist(),
            'mean_forecast': float(np.mean(forecast_result['forecast'])),
            'trend': 'increasing' if forecast_result['forecast'][-1] > forecast_result['forecast'][0] else 'decreasing'
        }
