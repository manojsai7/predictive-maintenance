"""Data quality checks for sensor data."""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class DataQualityReport:
    """Report containing data quality metrics."""
    missing_values: Dict[str, float]
    outliers: Dict[str, int]
    duplicate_rows: int
    data_types_valid: bool
    quality_score: float
    issues: List[str]


class DataQualityChecker:
    """Perform data quality checks on sensor data."""
    
    def __init__(self, outlier_std: float = 3.0):
        """
        Initialize data quality checker.
        
        Args:
            outlier_std: Number of standard deviations for outlier detection
        """
        self.outlier_std = outlier_std
    
    def check_missing_values(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Check for missing values in dataframe.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dictionary with column names and percentage of missing values
        """
        missing_pct = (df.isnull().sum() / len(df) * 100).to_dict()
        return {col: pct for col, pct in missing_pct.items() if pct > 0}
    
    def detect_outliers(self, df: pd.DataFrame, 
                       numeric_cols: Optional[List[str]] = None) -> Dict[str, int]:
        """
        Detect outliers using z-score method.
        
        Args:
            df: Input dataframe
            numeric_cols: List of numeric columns to check
            
        Returns:
            Dictionary with column names and number of outliers
        """
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        outliers = {}
        for col in numeric_cols:
            if col in df.columns:
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                outliers[col] = int(np.sum(z_scores > self.outlier_std))
        
        return outliers
    
    def check_duplicates(self, df: pd.DataFrame) -> int:
        """
        Check for duplicate rows.
        
        Args:
            df: Input dataframe
            
        Returns:
            Number of duplicate rows
        """
        return df.duplicated().sum()
    
    def validate_data_types(self, df: pd.DataFrame, 
                           expected_types: Dict[str, type]) -> bool:
        """
        Validate that columns have expected data types.
        
        Args:
            df: Input dataframe
            expected_types: Dictionary mapping column names to expected types
            
        Returns:
            True if all types match, False otherwise
        """
        for col, expected_type in expected_types.items():
            if col not in df.columns:
                return False
            if not pd.api.types.is_dtype_equal(df[col].dtype, expected_type):
                return False
        return True
    
    def generate_quality_report(self, df: pd.DataFrame,
                               numeric_cols: Optional[List[str]] = None) -> DataQualityReport:
        """
        Generate comprehensive data quality report.
        
        Args:
            df: Input dataframe
            numeric_cols: List of numeric columns to check
            
        Returns:
            DataQualityReport object
        """
        missing = self.check_missing_values(df)
        outliers = self.detect_outliers(df, numeric_cols)
        duplicates = self.check_duplicates(df)
        
        issues = []
        
        # Check for critical issues
        if any(pct > 50 for pct in missing.values()):
            issues.append("CRITICAL: Some columns have >50% missing values")
        
        if duplicates > len(df) * 0.1:
            issues.append(f"WARNING: {duplicates} duplicate rows found")
        
        if any(count > len(df) * 0.1 for count in outliers.values()):
            issues.append("WARNING: High number of outliers detected")
        
        # Calculate quality score (0-100)
        score = 100.0
        score -= min(50, sum(missing.values()) / len(df.columns))  # Max 50 points
        score -= min(30, (duplicates / len(df)) * 100)  # Max 30 points
        score -= min(20, sum(outliers.values()) / len(df) * 100)  # Max 20 points
        score = max(0, score)
        
        return DataQualityReport(
            missing_values=missing,
            outliers=outliers,
            duplicate_rows=duplicates,
            data_types_valid=True,
            quality_score=score,
            issues=issues
        )


class DataValidator:
    """Validate sensor data against expected ranges and constraints."""
    
    def __init__(self, sensor_ranges: Optional[Dict[str, Tuple[float, float]]] = None):
        """
        Initialize data validator.
        
        Args:
            sensor_ranges: Dictionary mapping sensor names to (min, max) valid ranges
        """
        self.sensor_ranges = sensor_ranges or {}
    
    def validate_ranges(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Validate that sensor values are within expected ranges.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dictionary with number of out-of-range values per sensor
        """
        violations = {}
        
        for sensor, (min_val, max_val) in self.sensor_ranges.items():
            if sensor in df.columns:
                out_of_range = ((df[sensor] < min_val) | (df[sensor] > max_val)).sum()
                if out_of_range > 0:
                    violations[sensor] = int(out_of_range)
        
        return violations
    
    def validate_temporal_consistency(self, df: pd.DataFrame, 
                                     time_col: str = 'timestamp') -> bool:
        """
        Validate that timestamps are monotonically increasing.
        
        Args:
            df: Input dataframe
            time_col: Name of timestamp column
            
        Returns:
            True if timestamps are valid, False otherwise
        """
        if time_col not in df.columns:
            return False
        
        time_series = pd.to_datetime(df[time_col])
        return time_series.is_monotonic_increasing


from scipy import stats
