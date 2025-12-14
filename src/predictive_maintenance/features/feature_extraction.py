"""Feature extraction for sensor/telemetry data."""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from scipy import stats


class SensorFeatureExtractor:
    """Extract statistical and domain features from sensor data."""
    
    def __init__(self, window_size: int = 50):
        """
        Initialize feature extractor.
        
        Args:
            window_size: Size of rolling window for feature calculation
        """
        self.window_size = window_size
        
    def extract_statistical_features(self, df: pd.DataFrame, 
                                     sensor_cols: List[str]) -> pd.DataFrame:
        """
        Extract statistical features from sensor data.
        
        Args:
            df: Input dataframe with sensor readings
            sensor_cols: List of sensor column names
            
        Returns:
            DataFrame with statistical features
        """
        features = df.copy()
        
        for col in sensor_cols:
            # Rolling statistics
            features[f'{col}_mean'] = df[col].rolling(self.window_size).mean()
            features[f'{col}_std'] = df[col].rolling(self.window_size).std()
            features[f'{col}_min'] = df[col].rolling(self.window_size).min()
            features[f'{col}_max'] = df[col].rolling(self.window_size).max()
            features[f'{col}_median'] = df[col].rolling(self.window_size).median()
            
            # Rate of change
            features[f'{col}_diff'] = df[col].diff()
            features[f'{col}_pct_change'] = df[col].pct_change()
            
            # Higher order statistics
            features[f'{col}_skew'] = df[col].rolling(self.window_size).skew()
            features[f'{col}_kurtosis'] = df[col].rolling(self.window_size).kurt()
            
        return features
    
    def extract_frequency_features(self, df: pd.DataFrame, 
                                   sensor_cols: List[str]) -> pd.DataFrame:
        """
        Extract frequency domain features using FFT.
        
        Args:
            df: Input dataframe with sensor readings
            sensor_cols: List of sensor column names
            
        Returns:
            DataFrame with frequency features
        """
        features = df.copy()
        
        for col in sensor_cols:
            # FFT-based features
            values = df[col].values
            if len(values) >= self.window_size:
                fft = np.fft.fft(values[-self.window_size:])
                fft_magnitude = np.abs(fft)
                
                features[f'{col}_fft_mean'] = np.mean(fft_magnitude)
                features[f'{col}_fft_std'] = np.std(fft_magnitude)
                features[f'{col}_fft_max'] = np.max(fft_magnitude)
                features[f'{col}_dominant_freq'] = np.argmax(fft_magnitude)
            
        return features
    
    def extract_all_features(self, df: pd.DataFrame, 
                            sensor_cols: List[str]) -> pd.DataFrame:
        """
        Extract all features from sensor data.
        
        Args:
            df: Input dataframe with sensor readings
            sensor_cols: List of sensor column names
            
        Returns:
            DataFrame with all features
        """
        features = self.extract_statistical_features(df, sensor_cols)
        features = self.extract_frequency_features(features, sensor_cols)
        
        # Drop NaN values created by rolling windows
        features = features.dropna()
        
        return features


class RULFeatureEngineer:
    """Feature engineering specifically for Remaining Useful Life (RUL) prediction."""
    
    def __init__(self):
        """Initialize RUL feature engineer."""
        pass
    
    def create_rul_labels(self, df: pd.DataFrame, 
                         failure_cycles: Dict[int, int]) -> pd.DataFrame:
        """
        Create RUL labels for training data.
        
        Args:
            df: Input dataframe with 'unit_id' and 'cycle' columns
            failure_cycles: Dictionary mapping unit_id to failure cycle
            
        Returns:
            DataFrame with RUL column
        """
        df_with_rul = df.copy()
        
        def calculate_rul(row):
            unit_id = row['unit_id']
            current_cycle = row['cycle']
            failure_cycle = failure_cycles.get(unit_id, current_cycle)
            return max(0, failure_cycle - current_cycle)
        
        df_with_rul['RUL'] = df_with_rul.apply(calculate_rul, axis=1)
        
        return df_with_rul
    
    def create_degradation_features(self, df: pd.DataFrame, 
                                   sensor_cols: List[str]) -> pd.DataFrame:
        """
        Create features that capture equipment degradation over time.
        
        Args:
            df: Input dataframe with sensor data
            sensor_cols: List of sensor column names
            
        Returns:
            DataFrame with degradation features
        """
        features = df.copy()
        
        # Group by unit and calculate cumulative features
        if 'unit_id' in df.columns:
            for col in sensor_cols:
                # Cumulative sum (total wear indicator)
                features[f'{col}_cumsum'] = df.groupby('unit_id')[col].cumsum()
                
                # Exponential weighted average (recent trend)
                features[f'{col}_ewm'] = df.groupby('unit_id')[col].transform(
                    lambda x: x.ewm(span=10).mean()
                )
                
                # Deviation from initial baseline
                features[f'{col}_deviation'] = df.groupby('unit_id')[col].transform(
                    lambda x: x - x.iloc[0] if len(x) > 0 else 0
                )
        
        return features
