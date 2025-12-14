# Predictive Maintenance Toolkit - API Reference

## Quick Reference

### DataProcessor
```python
from predictive_maintenance import DataProcessor

processor = DataProcessor()

# Complete preprocessing pipeline
processed = processor.prepare_data(
    data=df,
    sensor_columns=['temp', 'vibration'],
    normalize=True,
    remove_outliers=True
)

# Individual operations
clean = processor.handle_missing_values(df, strategy='interpolate')
normalized = processor.normalize_features(df)
featured = processor.extract_features(df, ['temp', 'vibration'])
```

### HealthScorer
```python
from predictive_maintenance import HealthScorer

scorer = HealthScorer()

# Setup
scorer.set_baseline(baseline_data, ['temp', 'vibration', 'pressure'])
scorer.set_weights({'temp': 0.4, 'vibration': 0.3, 'pressure': 0.3})

# Calculate health
health = scorer.calculate_overall_health(current_data)
category = scorer.get_health_category(health)  # excellent/good/fair/poor/critical
trend = scorer.get_health_trend(health_history)  # improving/stable/degrading
```

### AnomalyDetector
```python
from predictive_maintenance import AnomalyDetector

# Methods: 'isolation_forest', 'elliptic_envelope', 'statistical'
detector = AnomalyDetector(method='isolation_forest', contamination=0.1)

# Train and detect
detector.fit(training_data)
labels = detector.detect(test_data)  # -1 for anomaly, 1 for normal
summary = detector.get_anomaly_summary(test_data)
```

### RULPredictor
```python
from predictive_maintenance import RULPredictor

# Models: 'random_forest', 'gradient_boosting', 'linear'
predictor = RULPredictor(model_type='random_forest')

# Train
predictor.fit(training_data, rul_column='rul')

# Predict with confidence
result = predictor.predict_with_confidence(test_data)
rul = result['predictions']
lower = result['lower_bound']
upper = result['upper_bound']
```

### FailurePredictor
```python
from predictive_maintenance import FailurePredictor

# Models: 'random_forest', 'gradient_boosting', 'logistic'
predictor = FailurePredictor(model_type='random_forest')

# Train
predictor.fit(training_data, failure_column='failure')

# Predict with risk levels
result = predictor.predict_with_confidence(test_data)
probabilities = result['probabilities']  # 0-1 failure probability
risk_levels = result['risk_level']  # 0=Low, 1=Medium, 2=High, 3=Critical
```

### ForecastingEngine
```python
from predictive_maintenance import ForecastingEngine

engine = ForecastingEngine(forecast_horizon=20)

# Train
engine.fit(historical_data, target_columns=['temperature', 'vibration'])

# Forecast
forecast = engine.forecast_with_confidence(
    recent_data,
    column='temperature',
    steps=20
)
```

### AlertManager
```python
from predictive_maintenance import AlertManager, AlertSeverity

manager = AlertManager()

# Check conditions and generate alerts
manager.check_health_alerts(health_score, 'EQUIP-001')
manager.check_anomaly_alerts(anomaly_count, total_samples, 'EQUIP-001')
manager.check_failure_alerts(failure_prob, time_to_failure, 'EQUIP-001')
manager.check_rul_alerts(rul, threshold_warning=100, threshold_critical=50)

# Manage alerts
active = manager.get_active_alerts()
manager.acknowledge_alert(alert_id)
summary = manager.get_alert_summary()
```

## Common Workflows

### Workflow 1: Real-time Health Monitoring
```python
# 1. Process incoming data
processed = processor.prepare_data(sensor_data, sensor_columns)

# 2. Calculate health score
health = scorer.calculate_overall_health(current_reading)

# 3. Generate alerts
alert_manager.check_health_alerts(health, equipment_id)
```

### Workflow 2: Predictive Maintenance
```python
# 1. Detect anomalies
summary = detector.get_anomaly_summary(recent_data)

# 2. Predict RUL
rul = rul_predictor.predict(current_data)

# 3. Predict failure probability
failure_prob = failure_predictor.predict_proba(current_data)

# 4. Generate appropriate alerts
alert_manager.check_anomaly_alerts(summary['anomaly_count'], len(recent_data))
alert_manager.check_rul_alerts(rul[0])
alert_manager.check_failure_alerts(failure_prob[0])
```

### Workflow 3: Batch Analysis
```python
# Process batch of historical data
health_scores = scorer.calculate_health_batch(historical_data)
trend = scorer.get_health_trend(health_scores)

# Analyze anomalies
anomaly_labels = detector.detect(historical_data)
anomaly_rate = (anomaly_labels == -1).sum() / len(anomaly_labels)

# Forecast future state
forecast = engine.forecast_dataframe(
    historical_data,
    columns=['temperature', 'vibration'],
    steps=30
)
```

## Return Value Reference

### Health Scores
- **Health Score**: 0-100 (0=critical, 100=excellent)
- **Category**: 'excellent', 'good', 'fair', 'poor', 'critical'
- **Trend**: 'improving', 'stable', 'degrading'

### Anomaly Detection
- **Labels**: -1 (anomaly), 1 (normal)
- **Scores**: Higher = more anomalous
- **Anomaly Rate**: 0-1 (proportion of anomalies)

### Failure Prediction
- **Probability**: 0-1 (0=no failure, 1=certain failure)
- **Risk Level**: 0 (Low), 1 (Medium), 2 (High), 3 (Critical)

### Alert Severity
- **INFO**: Informational message
- **WARNING**: Potential issue (30-60% failure probability, 60-75 health)
- **CRITICAL**: Serious issue (60-80% failure probability, 40-60 health, >20% anomalies)
- **EMERGENCY**: Immediate action required (>80% failure probability, <40 health)

## Configuration Tips

### Anomaly Detection
- Use `contamination=0.05-0.15` for normal scenarios
- Use `method='statistical'` for interpretable results
- Use `method='isolation_forest'` for complex patterns

### Model Selection
- **Random Forest**: Best for general use, handles non-linearity
- **Gradient Boosting**: Best for complex patterns, slower training
- **Linear/Logistic**: Fast, interpretable, works for linear relationships

### Feature Engineering
- Use `windows=[5, 10, 20]` for short, medium, long-term patterns
- Use `lags=[1, 2, 3]` to capture recent history
- Normalize features when combining different scales

### Health Scoring
- Set baseline from at least 50-100 samples of normal operation
- Adjust weights based on domain knowledge
- Update baseline periodically to account for normal drift
