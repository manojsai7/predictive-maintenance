"""Alert management module for predictive maintenance."""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = 1
    WARNING = 2
    CRITICAL = 3
    EMERGENCY = 4


class AlertManager:
    """Manage alerts and early warnings for equipment issues."""
    
    def __init__(self):
        """Initialize the AlertManager."""
        self.alerts = []
        self.alert_rules = {}
        self.alert_history = []
    
    def add_alert_rule(
        self, 
        rule_name: str,
        condition_type: str,
        threshold: float,
        metric: str,
        severity: AlertSeverity = AlertSeverity.WARNING
    ):
        """
        Add an alert rule.
        
        Args:
            rule_name: Name of the rule
            condition_type: Type of condition ('greater', 'less', 'equal', 'between')
            threshold: Threshold value or tuple for 'between'
            metric: Metric to monitor
            severity: Alert severity level
        """
        self.alert_rules[rule_name] = {
            'condition_type': condition_type,
            'threshold': threshold,
            'metric': metric,
            'severity': severity
        }
    
    def check_alert_rule(
        self, 
        rule_name: str,
        value: float
    ) -> bool:
        """
        Check if a value triggers an alert rule.
        
        Args:
            rule_name: Name of the rule to check
            value: Current value to check
            
        Returns:
            True if alert should be triggered
        """
        if rule_name not in self.alert_rules:
            return False
        
        rule = self.alert_rules[rule_name]
        condition = rule['condition_type']
        threshold = rule['threshold']
        
        if condition == 'greater':
            return value > threshold
        elif condition == 'less':
            return value < threshold
        elif condition == 'equal':
            return abs(value - threshold) < 1e-6
        elif condition == 'between':
            if isinstance(threshold, (list, tuple)) and len(threshold) == 2:
                return not (threshold[0] <= value <= threshold[1])
        
        return False
    
    def create_alert(
        self, 
        alert_type: str,
        message: str,
        severity: AlertSeverity,
        data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a new alert.
        
        Args:
            alert_type: Type of alert
            message: Alert message
            severity: Alert severity
            data: Additional alert data
            
        Returns:
            Alert dictionary
        """
        alert = {
            'id': len(self.alerts),
            'type': alert_type,
            'message': message,
            'severity': severity.name,
            'timestamp': datetime.now().isoformat(),
            'data': data or {},
            'acknowledged': False
        }
        
        self.alerts.append(alert)
        self.alert_history.append(alert.copy())
        
        return alert
    
    def check_health_alerts(
        self, 
        health_score: float,
        equipment_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Check if health score triggers an alert.
        
        Args:
            health_score: Current health score
            equipment_id: Optional equipment identifier
            
        Returns:
            Alert dictionary if triggered, None otherwise
        """
        if health_score < 40:
            return self.create_alert(
                alert_type='health_critical',
                message=f'Equipment health is critical: {health_score:.1f}%',
                severity=AlertSeverity.CRITICAL,
                data={
                    'health_score': health_score,
                    'equipment_id': equipment_id
                }
            )
        elif health_score < 60:
            return self.create_alert(
                alert_type='health_warning',
                message=f'Equipment health is poor: {health_score:.1f}%',
                severity=AlertSeverity.WARNING,
                data={
                    'health_score': health_score,
                    'equipment_id': equipment_id
                }
            )
        
        return None
    
    def check_anomaly_alerts(
        self, 
        anomaly_count: int,
        total_samples: int,
        equipment_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Check if anomaly detection triggers an alert.
        
        Args:
            anomaly_count: Number of anomalies detected
            total_samples: Total number of samples
            equipment_id: Optional equipment identifier
            
        Returns:
            Alert dictionary if triggered, None otherwise
        """
        anomaly_rate = anomaly_count / total_samples if total_samples > 0 else 0
        
        if anomaly_rate > 0.2:
            return self.create_alert(
                alert_type='anomaly_critical',
                message=f'High anomaly rate detected: {anomaly_rate*100:.1f}%',
                severity=AlertSeverity.CRITICAL,
                data={
                    'anomaly_count': anomaly_count,
                    'total_samples': total_samples,
                    'anomaly_rate': anomaly_rate,
                    'equipment_id': equipment_id
                }
            )
        elif anomaly_rate > 0.1:
            return self.create_alert(
                alert_type='anomaly_warning',
                message=f'Elevated anomaly rate: {anomaly_rate*100:.1f}%',
                severity=AlertSeverity.WARNING,
                data={
                    'anomaly_count': anomaly_count,
                    'total_samples': total_samples,
                    'anomaly_rate': anomaly_rate,
                    'equipment_id': equipment_id
                }
            )
        
        return None
    
    def check_failure_alerts(
        self, 
        failure_probability: float,
        time_to_failure: Optional[float] = None,
        equipment_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Check if failure prediction triggers an alert.
        
        Args:
            failure_probability: Predicted failure probability
            time_to_failure: Estimated time until failure
            equipment_id: Optional equipment identifier
            
        Returns:
            Alert dictionary if triggered, None otherwise
        """
        if failure_probability > 0.8:
            return self.create_alert(
                alert_type='failure_imminent',
                message=f'Failure imminent: {failure_probability*100:.1f}% probability',
                severity=AlertSeverity.EMERGENCY,
                data={
                    'failure_probability': failure_probability,
                    'time_to_failure': time_to_failure,
                    'equipment_id': equipment_id
                }
            )
        elif failure_probability > 0.6:
            return self.create_alert(
                alert_type='failure_likely',
                message=f'Failure likely: {failure_probability*100:.1f}% probability',
                severity=AlertSeverity.CRITICAL,
                data={
                    'failure_probability': failure_probability,
                    'time_to_failure': time_to_failure,
                    'equipment_id': equipment_id
                }
            )
        elif failure_probability > 0.3:
            return self.create_alert(
                alert_type='failure_possible',
                message=f'Failure possible: {failure_probability*100:.1f}% probability',
                severity=AlertSeverity.WARNING,
                data={
                    'failure_probability': failure_probability,
                    'time_to_failure': time_to_failure,
                    'equipment_id': equipment_id
                }
            )
        
        return None
    
    def check_rul_alerts(
        self, 
        remaining_useful_life: float,
        threshold_warning: float = 100,
        threshold_critical: float = 50,
        equipment_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Check if RUL prediction triggers an alert.
        
        Args:
            remaining_useful_life: Predicted RUL value
            threshold_warning: RUL threshold for warning
            threshold_critical: RUL threshold for critical alert
            equipment_id: Optional equipment identifier
            
        Returns:
            Alert dictionary if triggered, None otherwise
        """
        if remaining_useful_life < threshold_critical:
            return self.create_alert(
                alert_type='rul_critical',
                message=f'Remaining useful life critical: {remaining_useful_life:.1f} time units',
                severity=AlertSeverity.CRITICAL,
                data={
                    'remaining_useful_life': remaining_useful_life,
                    'equipment_id': equipment_id
                }
            )
        elif remaining_useful_life < threshold_warning:
            return self.create_alert(
                alert_type='rul_warning',
                message=f'Remaining useful life low: {remaining_useful_life:.1f} time units',
                severity=AlertSeverity.WARNING,
                data={
                    'remaining_useful_life': remaining_useful_life,
                    'equipment_id': equipment_id
                }
            )
        
        return None
    
    def get_active_alerts(
        self, 
        severity: Optional[AlertSeverity] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all active (unacknowledged) alerts.
        
        Args:
            severity: Filter by severity level (None for all)
            
        Returns:
            List of active alerts
        """
        active = [a for a in self.alerts if not a['acknowledged']]
        
        if severity:
            active = [a for a in active if a['severity'] == severity.name]
        
        return active
    
    def acknowledge_alert(self, alert_id: int):
        """
        Acknowledge an alert.
        
        Args:
            alert_id: ID of the alert to acknowledge
        """
        for alert in self.alerts:
            if alert['id'] == alert_id:
                alert['acknowledged'] = True
                break
    
    def clear_all_alerts(self):
        """Clear all active alerts."""
        self.alerts = []
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """
        Get summary of alert status.
        
        Returns:
            Dictionary with alert statistics
        """
        active_alerts = self.get_active_alerts()
        
        severity_counts = {
            'INFO': 0,
            'WARNING': 0,
            'CRITICAL': 0,
            'EMERGENCY': 0
        }
        
        for alert in active_alerts:
            severity = alert['severity']
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            'total_alerts': len(self.alerts),
            'active_alerts': len(active_alerts),
            'acknowledged_alerts': len(self.alerts) - len(active_alerts),
            'severity_counts': severity_counts,
            'alert_history_size': len(self.alert_history)
        }
