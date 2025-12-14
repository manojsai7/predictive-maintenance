"""Tests for data quality checks."""
import pytest
import numpy as np
import pandas as pd
from predictive_maintenance.data_quality.checks import (
    DataQualityChecker,
    DataValidator
)


def test_data_quality_checker_initialization():
    """Test DataQualityChecker initialization."""
    checker = DataQualityChecker(outlier_std=3.0)
    assert checker.outlier_std == 3.0


def test_check_missing_values():
    """Test missing value detection."""
    df = pd.DataFrame({
        'col1': [1, 2, np.nan, 4],
        'col2': [1, 2, 3, 4],
        'col3': [np.nan, np.nan, 3, 4]
    })
    
    checker = DataQualityChecker()
    missing = checker.check_missing_values(df)
    
    assert 'col1' in missing
    assert 'col3' in missing
    assert 'col2' not in missing
    assert missing['col1'] == 25.0  # 1 out of 4 = 25%
    assert missing['col3'] == 50.0  # 2 out of 4 = 50%


def test_detect_outliers():
    """Test outlier detection."""
    df = pd.DataFrame({
        'col1': [1, 2, 3, 4, 100],  # 100 is an outlier
        'col2': [10, 11, 12, 13, 14]  # No outliers
    })
    
    checker = DataQualityChecker(outlier_std=2.0)
    outliers = checker.detect_outliers(df, ['col1', 'col2'])
    
    assert 'col1' in outliers
    assert outliers['col1'] > 0


def test_check_duplicates():
    """Test duplicate detection."""
    df = pd.DataFrame({
        'col1': [1, 2, 2, 3],
        'col2': [1, 2, 2, 3]
    })
    
    checker = DataQualityChecker()
    duplicates = checker.check_duplicates(df)
    
    assert duplicates == 1  # One duplicate row


def test_generate_quality_report():
    """Test quality report generation."""
    df = pd.DataFrame({
        'col1': [1, 2, 3, 4, 5],
        'col2': [10, 11, 12, 13, 14]
    })
    
    checker = DataQualityChecker()
    report = checker.generate_quality_report(df)
    
    assert report.quality_score >= 0
    assert report.quality_score <= 100
    assert isinstance(report.missing_values, dict)
    assert isinstance(report.outliers, dict)


def test_data_validator_initialization():
    """Test DataValidator initialization."""
    sensor_ranges = {'sensor_1': (0, 100)}
    validator = DataValidator(sensor_ranges=sensor_ranges)
    assert 'sensor_1' in validator.sensor_ranges


def test_validate_ranges():
    """Test range validation."""
    df = pd.DataFrame({
        'sensor_1': [50, 75, 150, 200],  # 150 and 200 are out of range
        'sensor_2': [10, 20, 30, 40]
    })
    
    sensor_ranges = {'sensor_1': (0, 100), 'sensor_2': (0, 50)}
    validator = DataValidator(sensor_ranges=sensor_ranges)
    violations = validator.validate_ranges(df)
    
    assert 'sensor_1' in violations
    assert violations['sensor_1'] == 2


def test_validate_temporal_consistency():
    """Test temporal consistency validation."""
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=5, freq='H')
    })
    
    validator = DataValidator()
    is_valid = validator.validate_temporal_consistency(df, 'timestamp')
    
    assert is_valid is True
    
    # Test with non-monotonic timestamps
    df_invalid = pd.DataFrame({
        'timestamp': ['2024-01-01 00:00:00', '2024-01-01 02:00:00', '2024-01-01 01:00:00']
    })
    
    is_valid_invalid = validator.validate_temporal_consistency(df_invalid, 'timestamp')
    assert is_valid_invalid is False
