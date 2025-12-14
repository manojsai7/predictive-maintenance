"""Unit tests for the Data Processor module."""

import pytest
import numpy as np
import pandas as pd
from predictive_maintenance.data_processor import DataProcessor


def test_data_processor_initialization():
    """Test DataProcessor initialization."""
    processor = DataProcessor()
    assert processor.feature_columns == []
    assert processor.scaler_params == {}


def test_load_data():
    """Test data loading."""
    processor = DataProcessor()
    data = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    
    loaded = processor.load_data(data)
    assert len(loaded) == 3
    assert list(loaded.columns) == ['a', 'b']


def test_load_empty_data():
    """Test loading empty data raises error."""
    processor = DataProcessor()
    empty_data = pd.DataFrame()
    
    with pytest.raises(ValueError):
        processor.load_data(empty_data)


def test_handle_missing_values_interpolate():
    """Test missing value handling with interpolation."""
    processor = DataProcessor()
    data = pd.DataFrame({'a': [1, np.nan, 3, np.nan, 5]})
    
    result = processor.handle_missing_values(data, strategy='interpolate')
    assert result['a'].isna().sum() == 0
    assert result['a'].iloc[1] == 2.0


def test_normalize_features():
    """Test feature normalization."""
    processor = DataProcessor()
    data = pd.DataFrame({'a': [0, 5, 10], 'b': [100, 200, 300]})
    
    normalized = processor.normalize_features(data)
    assert normalized['a'].min() == 0.0
    assert normalized['a'].max() == 1.0
    assert normalized['b'].min() == 0.0
    assert normalized['b'].max() == 1.0


def test_create_rolling_features():
    """Test rolling feature creation."""
    processor = DataProcessor()
    data = pd.DataFrame({'a': range(20)})
    
    result = processor.create_rolling_features(data, ['a'], windows=[5])
    assert 'a_rolling_mean_5' in result.columns
    assert 'a_rolling_std_5' in result.columns


def test_create_lag_features():
    """Test lag feature creation."""
    processor = DataProcessor()
    data = pd.DataFrame({'a': range(10)})
    
    result = processor.create_lag_features(data, ['a'], lags=[1, 2])
    assert 'a_lag_1' in result.columns
    assert 'a_lag_2' in result.columns
    assert result['a_lag_1'].iloc[1] == 0


def test_extract_features():
    """Test comprehensive feature extraction."""
    processor = DataProcessor()
    data = pd.DataFrame({
        'sensor1': np.random.randn(50),
        'sensor2': np.random.randn(50)
    })
    
    result = processor.extract_features(data, ['sensor1', 'sensor2'])
    assert len(result.columns) > len(data.columns)
    assert 'sensor1_rolling_mean_5' in result.columns
    assert 'sensor1_lag_1' in result.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
