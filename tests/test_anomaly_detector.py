"""Unit tests for the Anomaly Detector module."""

import pytest
import numpy as np
import pandas as pd
from predictive_maintenance.anomaly_detector import AnomalyDetector


def test_anomaly_detector_initialization():
    """Test AnomalyDetector initialization."""
    detector = AnomalyDetector(method="isolation_forest", contamination=0.1)
    assert detector.method == "isolation_forest"
    assert detector.contamination == 0.1
    assert detector.model is not None


def test_fit_and_detect_isolation_forest():
    """Test fitting and detecting with isolation forest."""
    np.random.seed(42)
    detector = AnomalyDetector(method="isolation_forest", contamination=0.1)
    
    # Create normal data
    data = pd.DataFrame({
        'x': np.random.randn(100),
        'y': np.random.randn(100)
    })
    
    detector.fit(data)
    labels = detector.detect(data)
    
    assert len(labels) == 100
    assert set(labels).issubset({-1, 1})


def test_fit_and_detect_statistical():
    """Test fitting and detecting with statistical method."""
    np.random.seed(42)
    detector = AnomalyDetector(method="statistical")
    
    # Create data with clear outliers
    data = pd.DataFrame({
        'x': np.concatenate([np.random.randn(95), np.array([10, 10, 10, 10, 10])])
    })
    
    detector.fit(data)
    labels = detector.detect(data)
    
    # Should detect some anomalies
    assert np.sum(labels == -1) > 0


def test_detect_with_scores():
    """Test anomaly detection with scores."""
    np.random.seed(42)
    detector = AnomalyDetector(method="isolation_forest", contamination=0.1)
    
    data = pd.DataFrame({
        'x': np.random.randn(50),
        'y': np.random.randn(50)
    })
    
    detector.fit(data)
    result = detector.detect_with_scores(data)
    
    assert 'labels' in result
    assert 'scores' in result
    assert len(result['labels']) == 50
    assert len(result['scores']) == 50


def test_get_anomaly_indices():
    """Test getting anomaly indices."""
    np.random.seed(42)
    detector = AnomalyDetector(method="statistical")
    
    data = pd.DataFrame({
        'x': np.concatenate([np.zeros(95), np.array([100, 100, 100, 100, 100])])
    })
    
    detector.fit(data)
    indices = detector.get_anomaly_indices(data)
    
    assert len(indices) > 0
    assert isinstance(indices, np.ndarray)


def test_get_anomaly_summary():
    """Test anomaly summary generation."""
    np.random.seed(42)
    detector = AnomalyDetector(method="isolation_forest", contamination=0.1)
    
    data = pd.DataFrame({
        'x': np.random.randn(100),
        'y': np.random.randn(100)
    })
    
    detector.fit(data)
    summary = detector.get_anomaly_summary(data)
    
    assert 'total_samples' in summary
    assert 'anomaly_count' in summary
    assert 'anomaly_rate' in summary
    assert 'anomaly_indices' in summary
    assert summary['total_samples'] == 100


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
