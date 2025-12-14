"""
Example: Complete Predictive Maintenance Workflow

This example demonstrates how to use the predictive maintenance toolkit
to forecast equipment health, detect anomalies, predict failures, and
generate alerts for early issue detection.
"""

import numpy as np
import pandas as pd
from predictive_maintenance import (
    DataProcessor,
    HealthScorer,
    AnomalyDetector,
    RULPredictor,
    FailurePredictor,
    ForecastingEngine,
    AlertManager
)


def generate_sample_data(n_samples=1000):
    """Generate synthetic sensor data for demonstration."""
    np.random.seed(42)
    
    # Generate time series data
    time = np.arange(n_samples)
    
    # Simulate sensor readings with degradation over time
    temperature = 70 + 10 * np.sin(time / 50) + np.random.normal(0, 2, n_samples) + time * 0.01
    vibration = 5 + 2 * np.sin(time / 30) + np.random.normal(0, 0.5, n_samples) + time * 0.002
    pressure = 100 + 5 * np.sin(time / 40) + np.random.normal(0, 1, n_samples) - time * 0.005
    
    # Add some anomalies
    anomaly_indices = np.random.choice(n_samples, size=50, replace=False)
    temperature[anomaly_indices] += np.random.normal(20, 5, len(anomaly_indices))
    
    # Create RUL (Remaining Useful Life) - decreases over time
    rul = np.maximum(500 - time * 0.5 + np.random.normal(0, 10, n_samples), 0)
    
    # Create failure labels (failures occur when RUL is very low)
    failure = (rul < 50).astype(int)
    
    data = pd.DataFrame({
        'time': time,
        'temperature': temperature,
        'vibration': vibration,
        'pressure': pressure,
        'rul': rul,
        'failure': failure
    })
    
    return data


def main():
    """Run complete predictive maintenance workflow."""
    print("=" * 80)
    print("Predictive Maintenance Toolkit - Complete Example")
    print("=" * 80)
    
    # Generate sample data
    print("\n1. Generating sample sensor data...")
    data = generate_sample_data(1000)
    print(f"   Generated {len(data)} samples with columns: {list(data.columns)}")
    
    # Split data into train and test
    train_size = int(len(data) * 0.7)
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    sensor_columns = ['temperature', 'vibration', 'pressure']
    
    # Step 1: Data Processing
    print("\n2. Processing sensor data...")
    processor = DataProcessor()
    processed_train = processor.prepare_data(
        train_data, 
        sensor_columns=sensor_columns,
        normalize=False,  # Keep original scale for health scoring
        remove_outliers=False
    )
    print(f"   Processed training data shape: {processed_train.shape}")
    print(f"   Created {len(processor.feature_columns)} engineered features")
    
    # Step 2: Health Scoring
    print("\n3. Calculating equipment health scores...")
    health_scorer = HealthScorer()
    
    # Set baseline from early data (when equipment was healthy)
    baseline_data = train_data[:100]
    health_scorer.set_baseline(baseline_data, sensor_columns)
    
    # Set weights for different sensors
    health_scorer.set_weights({
        'temperature': 0.4,
        'vibration': 0.3,
        'pressure': 0.3
    })
    
    # Calculate health scores
    health_scores = health_scorer.calculate_health_batch(test_data[sensor_columns])
    current_health = health_scores[-1]
    health_category = health_scorer.get_health_category(current_health)
    health_trend = health_scorer.get_health_trend(health_scores)
    
    print(f"   Current health score: {current_health:.1f}/100")
    print(f"   Health category: {health_category}")
    print(f"   Health trend: {health_trend}")
    
    # Step 3: Anomaly Detection
    print("\n4. Detecting anomalies...")
    anomaly_detector = AnomalyDetector(method="isolation_forest", contamination=0.05)
    anomaly_detector.fit(train_data, sensor_columns)
    
    anomaly_summary = anomaly_detector.get_anomaly_summary(test_data[sensor_columns])
    print(f"   Detected {anomaly_summary['anomaly_count']} anomalies")
    print(f"   Anomaly rate: {anomaly_summary['anomaly_rate']*100:.1f}%")
    print(f"   Max anomaly score: {anomaly_summary['max_anomaly_score']:.3f}")
    
    # Step 4: RUL Prediction
    print("\n5. Predicting Remaining Useful Life (RUL)...")
    rul_predictor = RULPredictor(model_type="random_forest")
    rul_predictor.fit(train_data, rul_column='rul', feature_columns=sensor_columns)
    
    rul_result = rul_predictor.predict_with_confidence(test_data[sensor_columns])
    current_rul = rul_result['predictions'][-1]
    rul_confidence = rul_result['std'][-1]
    
    print(f"   Predicted RUL: {current_rul:.1f} time units")
    print(f"   Confidence interval: [{current_rul - 1.96*rul_confidence:.1f}, {current_rul + 1.96*rul_confidence:.1f}]")
    
    # Get feature importance
    feature_importance = rul_predictor.get_feature_importance()
    if feature_importance:
        print("   Top contributing factors:")
        for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]:
            print(f"     - {feature}: {importance:.3f}")
    
    # Step 5: Failure Prediction
    print("\n6. Predicting failure probability...")
    failure_predictor = FailurePredictor(model_type="random_forest")
    failure_predictor.fit(train_data, failure_column='failure', feature_columns=sensor_columns)
    
    failure_result = failure_predictor.predict_with_confidence(test_data[sensor_columns])
    current_failure_prob = failure_result['probabilities'][-1]
    current_risk_level = failure_result['risk_level'][-1]
    
    risk_labels = ['Low', 'Medium', 'High', 'Critical']
    print(f"   Failure probability: {current_failure_prob*100:.1f}%")
    print(f"   Risk level: {risk_labels[current_risk_level]}")
    
    failure_summary = failure_predictor.get_failure_risk_summary(test_data[sensor_columns])
    print(f"   High/Critical risk samples: {failure_summary['high_risk_count'] + failure_summary['critical_risk_count']}")
    
    # Step 6: Forecasting
    print("\n7. Forecasting future equipment state...")
    forecasting_engine = ForecastingEngine(forecast_horizon=20)
    forecasting_engine.fit(train_data[['temperature', 'vibration', 'pressure']], 
                          target_columns=['temperature'], 
                          lookback=10)
    
    forecast_summary = forecasting_engine.get_forecast_summary(
        test_data[['temperature']], 
        column='temperature',
        steps=20
    )
    print(f"   Forecasted temperature trend: {forecast_summary['trend']}")
    print(f"   Mean forecasted value: {forecast_summary['mean_forecast']:.2f}")
    
    # Step 7: Alert Management
    print("\n8. Generating alerts and warnings...")
    alert_manager = AlertManager()
    
    # Check for various alert conditions
    alert_manager.check_health_alerts(current_health, equipment_id="PUMP-001")
    alert_manager.check_anomaly_alerts(
        anomaly_summary['anomaly_count'],
        anomaly_summary['total_samples'],
        equipment_id="PUMP-001"
    )
    alert_manager.check_failure_alerts(
        current_failure_prob,
        time_to_failure=current_rul,
        equipment_id="PUMP-001"
    )
    alert_manager.check_rul_alerts(
        current_rul,
        threshold_warning=200,
        threshold_critical=100,
        equipment_id="PUMP-001"
    )
    
    # Display alert summary
    alert_summary = alert_manager.get_alert_summary()
    print(f"   Total active alerts: {alert_summary['active_alerts']}")
    print(f"   Severity breakdown:")
    for severity, count in alert_summary['severity_counts'].items():
        if count > 0:
            print(f"     - {severity}: {count}")
    
    # Show active alerts
    active_alerts = alert_manager.get_active_alerts()
    if active_alerts:
        print("\n   Active Alerts:")
        for alert in active_alerts:
            print(f"     [{alert['severity']}] {alert['message']}")
    
    # Step 8: Summary Report
    print("\n" + "=" * 80)
    print("PREDICTIVE MAINTENANCE SUMMARY REPORT")
    print("=" * 80)
    print(f"Equipment ID: PUMP-001")
    print(f"Timestamp: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nHealth Status:")
    print(f"  - Overall Health Score: {current_health:.1f}/100 ({health_category})")
    print(f"  - Health Trend: {health_trend}")
    print(f"\nPredictive Analytics:")
    print(f"  - Remaining Useful Life: {current_rul:.1f} time units")
    print(f"  - Failure Probability: {current_failure_prob*100:.1f}% ({risk_labels[current_risk_level]} risk)")
    print(f"  - Anomalies Detected: {anomaly_summary['anomaly_count']} ({anomaly_summary['anomaly_rate']*100:.1f}%)")
    print(f"\nRecommendations:")
    
    if current_health < 60:
        print("  ⚠ URGENT: Schedule immediate inspection and maintenance")
    elif current_failure_prob > 0.6:
        print("  ⚠ WARNING: High failure risk detected - plan preventive maintenance")
    elif current_rul < 100:
        print("  ⚠ ATTENTION: Low RUL - prepare for equipment replacement/overhaul")
    elif anomaly_summary['anomaly_rate'] > 0.1:
        print("  ⚠ NOTICE: Elevated anomaly rate - investigate sensor readings")
    else:
        print("  ✓ Equipment operating within normal parameters - continue monitoring")
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
