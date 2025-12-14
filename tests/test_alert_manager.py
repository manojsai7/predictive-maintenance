"""Unit tests for the Alert Manager module."""

import pytest
import pandas as pd
from predictive_maintenance.alert_manager import AlertManager, AlertSeverity


def test_alert_manager_initialization():
    """Test AlertManager initialization."""
    manager = AlertManager()
    assert manager.alerts == []
    assert manager.alert_rules == {}
    assert manager.alert_history == []


def test_add_alert_rule():
    """Test adding alert rules."""
    manager = AlertManager()
    manager.add_alert_rule(
        rule_name='temp_high',
        condition_type='greater',
        threshold=80,
        metric='temperature',
        severity=AlertSeverity.WARNING
    )
    
    assert 'temp_high' in manager.alert_rules
    assert manager.alert_rules['temp_high']['threshold'] == 80


def test_check_alert_rule():
    """Test checking alert rules."""
    manager = AlertManager()
    manager.add_alert_rule(
        rule_name='temp_high',
        condition_type='greater',
        threshold=80,
        metric='temperature'
    )
    
    assert manager.check_alert_rule('temp_high', 85) == True
    assert manager.check_alert_rule('temp_high', 75) == False


def test_create_alert():
    """Test alert creation."""
    manager = AlertManager()
    alert = manager.create_alert(
        alert_type='test',
        message='Test alert',
        severity=AlertSeverity.WARNING
    )
    
    assert alert['type'] == 'test'
    assert alert['message'] == 'Test alert'
    assert alert['severity'] == 'WARNING'
    assert alert['acknowledged'] == False
    assert len(manager.alerts) == 1


def test_check_health_alerts():
    """Test health-based alert generation."""
    manager = AlertManager()
    
    # Critical health should generate alert
    alert = manager.check_health_alerts(35, 'PUMP-001')
    assert alert is not None
    assert alert['severity'] == 'CRITICAL'
    
    # Good health should not generate alert
    alert = manager.check_health_alerts(85, 'PUMP-001')
    assert alert is None


def test_check_anomaly_alerts():
    """Test anomaly-based alert generation."""
    manager = AlertManager()
    
    # High anomaly rate should generate alert
    alert = manager.check_anomaly_alerts(25, 100, 'PUMP-001')
    assert alert is not None
    assert alert['severity'] == 'CRITICAL'
    
    # Low anomaly rate should not generate alert
    alert = manager.check_anomaly_alerts(5, 100, 'PUMP-001')
    assert alert is None


def test_check_failure_alerts():
    """Test failure prediction alerts."""
    manager = AlertManager()
    
    # High failure probability should generate emergency alert
    alert = manager.check_failure_alerts(0.85, 10, 'PUMP-001')
    assert alert is not None
    assert alert['severity'] == 'EMERGENCY'
    
    # Low failure probability should not generate alert
    alert = manager.check_failure_alerts(0.2, 100, 'PUMP-001')
    assert alert is None


def test_check_rul_alerts():
    """Test RUL-based alerts."""
    manager = AlertManager()
    
    # Low RUL should generate critical alert
    alert = manager.check_rul_alerts(30, 100, 50, 'PUMP-001')
    assert alert is not None
    assert alert['severity'] == 'CRITICAL'
    
    # High RUL should not generate alert
    alert = manager.check_rul_alerts(200, 100, 50, 'PUMP-001')
    assert alert is None


def test_get_active_alerts():
    """Test getting active alerts."""
    manager = AlertManager()
    
    manager.create_alert('test1', 'Message 1', AlertSeverity.WARNING)
    manager.create_alert('test2', 'Message 2', AlertSeverity.CRITICAL)
    
    active = manager.get_active_alerts()
    assert len(active) == 2
    
    # Test filtering by severity
    critical = manager.get_active_alerts(severity=AlertSeverity.CRITICAL)
    assert len(critical) == 1


def test_acknowledge_alert():
    """Test alert acknowledgment."""
    manager = AlertManager()
    
    alert = manager.create_alert('test', 'Test', AlertSeverity.WARNING)
    alert_id = alert['id']
    
    manager.acknowledge_alert(alert_id)
    
    active = manager.get_active_alerts()
    assert len(active) == 0


def test_clear_all_alerts():
    """Test clearing all alerts."""
    manager = AlertManager()
    
    manager.create_alert('test1', 'Message 1', AlertSeverity.WARNING)
    manager.create_alert('test2', 'Message 2', AlertSeverity.CRITICAL)
    
    manager.clear_all_alerts()
    assert len(manager.alerts) == 0


def test_get_alert_summary():
    """Test alert summary generation."""
    manager = AlertManager()
    
    manager.create_alert('test1', 'Message 1', AlertSeverity.WARNING)
    manager.create_alert('test2', 'Message 2', AlertSeverity.CRITICAL)
    
    summary = manager.get_alert_summary()
    
    assert summary['total_alerts'] == 2
    assert summary['active_alerts'] == 2
    assert summary['severity_counts']['WARNING'] == 1
    assert summary['severity_counts']['CRITICAL'] == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
