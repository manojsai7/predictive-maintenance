"""
Quick Start Guide for Predictive Maintenance Toolkit
"""

# ============================================================================
# BASIC USAGE PATTERNS
# ============================================================================

import pandas as pd
import numpy as np
from predictive_maintenance import (
    DataProcessor, HealthScorer, AnomalyDetector,
    RULPredictor, FailurePredictor, AlertManager
)

# ----------------------------------------------------------------------------
# 1. SIMPLE HEALTH MONITORING
# ----------------------------------------------------------------------------
print("Example 1: Simple Health Monitoring")
print("-" * 50)

# Load your sensor data
sensor_data = pd.DataFrame({
    'temperature': [70, 72, 71, 75, 78, 82, 85],
    'vibration': [5.0, 5.2, 5.1, 5.5, 6.0, 6.5, 7.0],
    'pressure': [100, 99, 98, 96, 94, 92, 90]
})

# Initialize health scorer
health_scorer = HealthScorer()

# Set baseline from healthy operation (first few samples)
baseline = sensor_data.head(3)
health_scorer.set_baseline(baseline, ['temperature', 'vibration', 'pressure'])

# Calculate current health
current_reading = sensor_data.iloc[-1]
health_score = health_scorer.calculate_overall_health(current_reading)
health_category = health_scorer.get_health_category(health_score)

print(f"Health Score: {health_score:.1f}/100")
print(f"Status: {health_category}")
print()

# ----------------------------------------------------------------------------
# 2. ANOMALY DETECTION IN REAL-TIME
# ----------------------------------------------------------------------------
print("Example 2: Real-time Anomaly Detection")
print("-" * 50)

# Train anomaly detector on normal data
normal_data = sensor_data.head(5)
detector = AnomalyDetector(method="statistical")
detector.fit(normal_data, ['temperature', 'vibration', 'pressure'])

# Check for anomalies in new data
test_data = sensor_data.tail(2)
summary = detector.get_anomaly_summary(test_data)

print(f"Samples analyzed: {summary['total_samples']}")
print(f"Anomalies found: {summary['anomaly_count']}")
print(f"Anomaly rate: {summary['anomaly_rate']*100:.1f}%")
print()

# ----------------------------------------------------------------------------
# 3. FAILURE PREDICTION WITH ALERTS
# ----------------------------------------------------------------------------
print("Example 3: Failure Prediction with Alerts")
print("-" * 50)

# Simulate training data with failure labels
np.random.seed(42)
train_size = 100
train_data = pd.DataFrame({
    'temperature': np.random.normal(70, 5, train_size),
    'vibration': np.random.normal(5, 1, train_size),
    'pressure': np.random.normal(100, 3, train_size),
    'failure': np.random.choice([0, 1], train_size, p=[0.9, 0.1])
})

# Train failure predictor
predictor = FailurePredictor(model_type="logistic")
predictor.fit(train_data, 'failure', ['temperature', 'vibration', 'pressure'])

# Predict failure for current state
current = sensor_data.iloc[-1:][['temperature', 'vibration', 'pressure']]
result = predictor.predict_with_confidence(current)
failure_prob = result['probabilities'][0]

print(f"Failure Probability: {failure_prob*100:.1f}%")
print(f"Risk Level: {['Low', 'Medium', 'High', 'Critical'][result['risk_level'][0]]}")

# Generate alerts based on prediction
alert_manager = AlertManager()
alert = alert_manager.check_failure_alerts(failure_prob, equipment_id="PUMP-001")
if alert:
    print(f"Alert Generated: {alert['message']}")
print()

# ----------------------------------------------------------------------------
# 4. COMPLETE WORKFLOW WITH DATA PROCESSING
# ----------------------------------------------------------------------------
print("Example 4: Complete Workflow")
print("-" * 50)

# Process raw sensor data
processor = DataProcessor()
processed = processor.prepare_data(
    sensor_data,
    sensor_columns=['temperature', 'vibration', 'pressure'],
    normalize=False,
    remove_outliers=False
)

print(f"Original features: {len(sensor_data.columns)}")
print(f"After processing: {len(processed.columns)}")
print(f"New features added: {len(processor.feature_columns)}")
print()

# ----------------------------------------------------------------------------
# 5. BATCH ANALYSIS
# ----------------------------------------------------------------------------
print("Example 5: Batch Health Analysis")
print("-" * 50)

# Analyze health scores over time
health_scores = health_scorer.calculate_health_batch(
    sensor_data[['temperature', 'vibration', 'pressure']]
)

# Determine trend
trend = health_scorer.get_health_trend(health_scores)
print(f"Average Health: {health_scores.mean():.1f}/100")
print(f"Current Health: {health_scores[-1]:.1f}/100")
print(f"Trend: {trend}")
print()

# ----------------------------------------------------------------------------
# 6. ALERT MANAGEMENT
# ----------------------------------------------------------------------------
print("Example 6: Alert Management")
print("-" * 50)

# Check multiple alert conditions
alert_manager = AlertManager()

# Health alerts
alert_manager.check_health_alerts(health_score, "EQUIPMENT-001")

# Anomaly alerts
alert_manager.check_anomaly_alerts(
    summary['anomaly_count'],
    summary['total_samples'],
    "EQUIPMENT-001"
)

# Get alert summary
alert_summary = alert_manager.get_alert_summary()
print(f"Active Alerts: {alert_summary['active_alerts']}")
print(f"By Severity:")
for severity, count in alert_summary['severity_counts'].items():
    if count > 0:
        print(f"  {severity}: {count}")

# Display alerts
active_alerts = alert_manager.get_active_alerts()
if active_alerts:
    print("\nAlert Details:")
    for alert in active_alerts:
        print(f"  [{alert['severity']}] {alert['message']}")

print("\n" + "=" * 50)
print("Quick start examples completed!")
print("=" * 50)
