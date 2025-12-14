"""Unit tests for the Health Scorer module."""

import pytest
import numpy as np
import pandas as pd
from predictive_maintenance.health_scorer import HealthScorer


def test_health_scorer_initialization():
    """Test HealthScorer initialization."""
    scorer = HealthScorer()
    assert scorer.baseline_values == {}
    assert scorer.weights == {}
    assert 'excellent' in scorer.thresholds


def test_set_baseline():
    """Test baseline setting."""
    scorer = HealthScorer()
    data = pd.DataFrame({
        'temp': [70, 71, 69, 70, 71],
        'vibration': [5, 5.1, 4.9, 5, 5.1]
    })
    
    scorer.set_baseline(data, ['temp', 'vibration'])
    assert 'temp' in scorer.baseline_values
    assert 'vibration' in scorer.baseline_values
    assert 'mean' in scorer.baseline_values['temp']


def test_set_weights():
    """Test weight setting."""
    scorer = HealthScorer()
    weights = {'temp': 0.6, 'vibration': 0.4}
    
    scorer.set_weights(weights)
    assert scorer.weights['temp'] == 0.6
    assert scorer.weights['vibration'] == 0.4


def test_calculate_metric_health():
    """Test single metric health calculation."""
    scorer = HealthScorer()
    data = pd.DataFrame({'temp': [70, 70, 70, 70, 70]})
    scorer.set_baseline(data, ['temp'])
    
    # Value at baseline should have high health
    health = scorer.calculate_metric_health(70, 'temp')
    assert health == 100.0
    
    # Unknown metric should return default
    health = scorer.calculate_metric_health(50, 'unknown')
    assert health == 100.0


def test_calculate_overall_health():
    """Test overall health calculation."""
    scorer = HealthScorer()
    data = pd.DataFrame({
        'temp': [70, 70, 70, 70, 70],
        'vibration': [5, 5, 5, 5, 5]
    })
    scorer.set_baseline(data, ['temp', 'vibration'])
    scorer.set_weights({'temp': 0.5, 'vibration': 0.5})
    
    current = pd.Series({'temp': 70, 'vibration': 5})
    health = scorer.calculate_overall_health(current)
    assert health == 100.0


def test_get_health_category():
    """Test health category classification."""
    scorer = HealthScorer()
    
    assert scorer.get_health_category(95) == 'excellent'
    assert scorer.get_health_category(80) == 'good'
    assert scorer.get_health_category(65) == 'fair'
    assert scorer.get_health_category(50) == 'poor'
    assert scorer.get_health_category(30) == 'critical'


def test_get_health_trend():
    """Test health trend detection."""
    scorer = HealthScorer()
    
    # Improving trend
    improving = np.array([50, 55, 60, 65, 70])
    assert scorer.get_health_trend(improving) == 'improving'
    
    # Degrading trend
    degrading = np.array([70, 65, 60, 55, 50])
    assert scorer.get_health_trend(degrading) == 'degrading'
    
    # Stable trend
    stable = np.array([70, 70.5, 70, 70.5, 70])
    assert scorer.get_health_trend(stable) == 'stable'


def test_get_health_report():
    """Test comprehensive health report generation."""
    scorer = HealthScorer()
    data = pd.DataFrame({
        'temp': [70, 70, 70],
        'vibration': [5, 5, 5]
    })
    scorer.set_baseline(data, ['temp', 'vibration'])
    
    current = pd.Series({'temp': 72, 'vibration': 5.5})
    report = scorer.get_health_report(current, ['temp', 'vibration'])
    
    assert 'overall_health_score' in report
    assert 'health_category' in report
    assert 'metric_scores' in report
    assert 'timestamp' in report


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
