"""Tests for feature extraction module."""
import pytest
import numpy as np
import pandas as pd
from predictive_maintenance.features.feature_extraction import (
    SensorFeatureExtractor, 
    RULFeatureEngineer
)


def test_sensor_feature_extractor_initialization():
    """Test SensorFeatureExtractor initialization."""
    extractor = SensorFeatureExtractor(window_size=50)
    assert extractor.window_size == 50


def test_extract_statistical_features():
    """Test statistical feature extraction."""
    # Create sample data
    df = pd.DataFrame({
        'sensor_1': np.random.randn(100),
        'sensor_2': np.random.randn(100)
    })
    
    extractor = SensorFeatureExtractor(window_size=10)
    features = extractor.extract_statistical_features(df, ['sensor_1', 'sensor_2'])
    
    # Check that new features are created
    assert 'sensor_1_mean' in features.columns
    assert 'sensor_1_std' in features.columns
    assert 'sensor_2_mean' in features.columns


def test_rul_feature_engineer_initialization():
    """Test RULFeatureEngineer initialization."""
    engineer = RULFeatureEngineer()
    assert engineer is not None


def test_create_rul_labels():
    """Test RUL label creation."""
    df = pd.DataFrame({
        'unit_id': [1, 1, 1, 2, 2, 2],
        'cycle': [1, 2, 3, 1, 2, 3]
    })
    
    failure_cycles = {1: 3, 2: 3}
    
    engineer = RULFeatureEngineer()
    df_with_rul = engineer.create_rul_labels(df, failure_cycles)
    
    assert 'RUL' in df_with_rul.columns
    assert df_with_rul.loc[0, 'RUL'] == 2  # cycle 1, fails at 3
    assert df_with_rul.loc[2, 'RUL'] == 0  # cycle 3, fails at 3


def test_create_degradation_features():
    """Test degradation feature creation."""
    df = pd.DataFrame({
        'unit_id': [1, 1, 1, 1],
        'sensor_1': [50, 55, 60, 65],
        'sensor_2': [100, 105, 110, 115]
    })
    
    engineer = RULFeatureEngineer()
    features = engineer.create_degradation_features(df, ['sensor_1', 'sensor_2'])
    
    assert 'sensor_1_cumsum' in features.columns
    assert 'sensor_1_ewm' in features.columns
    assert 'sensor_1_deviation' in features.columns
