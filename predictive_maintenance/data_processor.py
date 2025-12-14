"""Data processing module for predictive maintenance."""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any
from scipy import stats


class DataProcessor:
    """Process and prepare sensor/equipment data for predictive maintenance."""
    
    def __init__(self):
        """Initialize the DataProcessor."""
        self.feature_columns = []
        self.scaler_params = {}
        
    def load_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Load and validate input data.
        
        Args:
            data: DataFrame containing sensor/equipment data
            
        Returns:
            Validated DataFrame
        """
        if data.empty:
            raise ValueError("Input data is empty")
        return data.copy()
    
    def handle_missing_values(
        self, 
        data: pd.DataFrame, 
        strategy: str = "interpolate"
    ) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            data: Input DataFrame
            strategy: Strategy for handling missing values 
                     ('interpolate', 'forward_fill', 'drop')
        
        Returns:
            DataFrame with handled missing values
        """
        if strategy == "interpolate":
            return data.interpolate(method='linear', limit_direction='both')
        elif strategy == "forward_fill":
            return data.ffill().bfill()
        elif strategy == "drop":
            return data.dropna()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def remove_outliers(
        self, 
        data: pd.DataFrame, 
        columns: Optional[List[str]] = None,
        threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        Remove outliers using z-score method.
        
        Args:
            data: Input DataFrame
            columns: Columns to check for outliers (None for all numeric)
            threshold: Z-score threshold for outlier detection
            
        Returns:
            DataFrame with outliers removed
        """
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        mask = np.abs(stats.zscore(data[columns], nan_policy='omit')) < threshold
        mask = mask.all(axis=1)
        return data[mask]
    
    def normalize_features(
        self, 
        data: pd.DataFrame, 
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Normalize features using min-max scaling.
        
        Args:
            data: Input DataFrame
            columns: Columns to normalize (None for all numeric)
            
        Returns:
            DataFrame with normalized features
        """
        result = data.copy()
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            min_val = data[col].min()
            max_val = data[col].max()
            self.scaler_params[col] = {'min': min_val, 'max': max_val}
            
            if max_val - min_val != 0:
                result[col] = (data[col] - min_val) / (max_val - min_val)
            else:
                result[col] = 0
        
        return result
    
    def create_rolling_features(
        self, 
        data: pd.DataFrame, 
        columns: List[str],
        windows: List[int] = [5, 10, 20]
    ) -> pd.DataFrame:
        """
        Create rolling window features for time series data.
        
        Args:
            data: Input DataFrame
            columns: Columns to create rolling features for
            windows: List of window sizes
            
        Returns:
            DataFrame with additional rolling features
        """
        result = data.copy()
        
        for col in columns:
            for window in windows:
                result[f'{col}_rolling_mean_{window}'] = \
                    data[col].rolling(window=window, min_periods=1).mean()
                result[f'{col}_rolling_std_{window}'] = \
                    data[col].rolling(window=window, min_periods=1).std()
        
        return result
    
    def create_lag_features(
        self, 
        data: pd.DataFrame, 
        columns: List[str],
        lags: List[int] = [1, 2, 3]
    ) -> pd.DataFrame:
        """
        Create lagged features for time series data.
        
        Args:
            data: Input DataFrame
            columns: Columns to create lag features for
            lags: List of lag values
            
        Returns:
            DataFrame with additional lag features
        """
        result = data.copy()
        
        for col in columns:
            for lag in lags:
                result[f'{col}_lag_{lag}'] = data[col].shift(lag)
        
        return result.bfill()
    
    def extract_features(
        self, 
        data: pd.DataFrame,
        sensor_columns: List[str]
    ) -> pd.DataFrame:
        """
        Extract comprehensive features from sensor data.
        
        Args:
            data: Input DataFrame with sensor readings
            sensor_columns: List of sensor column names
            
        Returns:
            DataFrame with extracted features
        """
        # Create rolling features
        result = self.create_rolling_features(data, sensor_columns)
        
        # Create lag features
        result = self.create_lag_features(result, sensor_columns)
        
        # Add rate of change features
        for col in sensor_columns:
            result[f'{col}_rate_of_change'] = data[col].diff()
        
        # Store feature column names
        self.feature_columns = [c for c in result.columns if c not in data.columns]
        
        return result
    
    def prepare_data(
        self, 
        data: pd.DataFrame,
        sensor_columns: List[str],
        normalize: bool = True,
        remove_outliers: bool = True
    ) -> pd.DataFrame:
        """
        Complete data preparation pipeline.
        
        Args:
            data: Input DataFrame
            sensor_columns: List of sensor column names
            normalize: Whether to normalize features
            remove_outliers: Whether to remove outliers
            
        Returns:
            Processed DataFrame ready for modeling
        """
        # Load and validate
        processed = self.load_data(data)
        
        # Handle missing values
        processed = self.handle_missing_values(processed)
        
        # Remove outliers if requested
        if remove_outliers:
            processed = self.remove_outliers(processed, sensor_columns)
        
        # Extract features
        processed = self.extract_features(processed, sensor_columns)
        
        # Normalize if requested
        if normalize:
            numeric_cols = processed.select_dtypes(include=[np.number]).columns.tolist()
            processed = self.normalize_features(processed, numeric_cols)
        
        return processed
