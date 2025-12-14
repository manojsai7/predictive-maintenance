
# Predictive Maintenance
Machine learning models and pipelines for equipment health monitoring and failure prediction.

## Features
- **Feature pipelines** for sensor/telemetry data processing
  - Statistical features (mean, std, min, max, etc.)
  - Frequency domain features (FFT-based)
  - Degradation indicators
- **Models**: Classical ML + Time-series Deep Learning
  - Random Forest and XGBoost for RUL prediction
  - LSTM networks for sequence-based prediction
  - Transformer models for time-series analysis
  - TSFresh integration for automatic feature extraction
- **Remaining Useful Life (RUL)** estimation
- **Drift detection** and data quality checks
  - Statistical drift detection (KS test)
  - Concept drift monitoring
  - Data quality reports
- **Model monitoring** dashboards
  - Performance tracking over time
  - Alert management
- **Batch and streaming inference**
  - FastAPI REST API
  - Batch prediction endpoints

## Tech Stack
- Python 3.8+
- scikit-learn, XGBoost
- PyTorch (Deep Learning)
- Pandas (Data processing)
- MLflow (Experiment tracking)
- FastAPI (Inference service)
- Docker + docker-compose

## Project Structure
```
predictive-maintenance/
├── src/
│   └── predictive_maintenance/
│       ├── features/          # Feature engineering
│       ├── models/            # ML models (classical & deep learning)
│       ├── inference/         # FastAPI inference service
│       ├── monitoring/        # Drift detection and monitoring
│       └── data_quality/      # Data quality checks
├── notebooks/                 # Jupyter notebooks for exploration
├── tests/                     # Unit tests
├── data/                      # Data directory
│   ├── raw/                   # Raw data
│   └── processed/             # Processed data
├── models/                    # Saved models
├── configs/                   # Configuration files
├── requirements.txt           # Python dependencies
├── setup.py                   # Package setup
├── Dockerfile                 # Docker configuration
└── docker-compose.yml         # Docker Compose configuration
```

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/manojsai7/predictive-maintenance
cd predictive-maintenance
```

### 2. Set up Python environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Install the package
```bash
pip install -e .
```

### 4. Generate synthetic data
```python
from predictive_maintenance.data_generator import SyntheticDataGenerator

generator = SyntheticDataGenerator(random_state=42)
train_df, test_df = generator.generate_complete_dataset(n_units=100)
generator.save_dataset(train_df, 'data/train_data.csv')
generator.save_dataset(test_df, 'data/test_data.csv')
```

### 5. Explore notebooks
Run Jupyter notebooks in `notebooks/` to explore sample data and train models:
```bash
jupyter notebook
```

## Usage Examples

### Feature Engineering
```python
from predictive_maintenance.features.feature_extraction import SensorFeatureExtractor

extractor = SensorFeatureExtractor(window_size=50)
features = extractor.extract_all_features(df, sensor_cols=['sensor_1', 'sensor_2'])
```

### Train a Model
```python
from predictive_maintenance.models.classical_models import RULPredictor

model = RULPredictor(model_type='random_forest')
model.build_model(n_estimators=100, max_depth=10)
model.train(X_train, y_train)
model.save_model('models/rul_predictor.pkl')
```

### Data Quality Checks
```python
from predictive_maintenance.data_quality.checks import DataQualityChecker

checker = DataQualityChecker()
report = checker.generate_quality_report(df)
print(f"Quality Score: {report.quality_score}")
```

### Drift Detection
```python
from predictive_maintenance.monitoring.drift_detection import DataDriftDetector

detector = DataDriftDetector()
detector.fit(reference_data)
drift_report = detector.detect_drift(current_data)
```

### Inference API
Start the FastAPI inference service:
```bash
uvicorn predictive_maintenance.inference.api:app --host 0.0.0.0 --port 8000
```

Or using Docker:
```bash
docker-compose up
```

Then make predictions:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "2024-01-01T00:00:00",
    "sensor_values": {"sensor_1": 75.5, "sensor_2": 82.3},
    "unit_id": 1,
    "cycle": 100
  }'
```

## Docker Deployment

### Build and run with Docker Compose
```bash
docker-compose up -d
```

This will start:
- **Inference API** on port 8000
- **MLflow** tracking server on port 5000
- **Jupyter Notebook** on port 8888

### Access services
- Inference API: http://localhost:8000
- MLflow UI: http://localhost:5000
- Jupyter Notebook: http://localhost:8888

## Testing

Run tests with pytest:
```bash
pytest tests/ -v
```

Run with coverage:
```bash
pytest tests/ --cov=predictive_maintenance --cov-report=html
```

## MLflow Integration

Track experiments with MLflow:
```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("predictive_maintenance")

with mlflow.start_run():
    model.train(X_train, y_train)
    mlflow.log_params({"n_estimators": 100, "max_depth": 10})
    mlflow.log_metrics({"rmse": rmse, "mae": mae})
    mlflow.sklearn.log_model(model, "model")
```

## Roadmap
- [x] Core feature engineering pipeline
- [x] Classical ML models (Random Forest, XGBoost)
- [x] Deep learning models (LSTM, Transformer)
- [x] Data quality checks
- [x] Drift detection
- [x] FastAPI inference service
- [x] Docker deployment
- [x] Synthetic data generator
- [ ] Example datasets (NASA Turbofan, etc.)
- [ ] Edge deployment guide
- [ ] AutoML recipes for baseline models
- [ ] Grafana/Prometheus monitoring example
- [ ] Streaming inference with Kafka
- [ ] Model explainability (SHAP values)

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
MIT License - see LICENSE file for details
=======
# Predictive Maintenance Toolkit

A comprehensive toolkit for forecasting equipment health, catching issues early, and reducing downtime with ML-driven insights.

## Overview

This toolkit provides a complete solution for predictive maintenance of industrial equipment. It combines multiple machine learning techniques to:

- **Forecast Equipment Health**: Calculate and track equipment health scores based on sensor data
- **Detect Anomalies**: Identify unusual patterns and behaviors in equipment operation
- **Predict Failures**: Estimate the probability of equipment failures before they occur
- **Estimate Remaining Useful Life (RUL)**: Predict how much operational time remains before maintenance is needed
- **Generate Early Warnings**: Automatically alert operators about potential issues
- **Forecast Future States**: Project equipment metrics into the future to plan maintenance activities

## Features

### 🔍 Data Processing
- Automated data cleaning and validation
- Missing value handling (interpolation, forward fill, drop)
- Outlier detection and removal
- Feature engineering (rolling statistics, lag features, rate of change)
- Feature normalization and scaling

### 📊 Health Scoring
- Baseline-based health calculation
- Weighted multi-metric health scores
- Health trend analysis (improving, stable, degrading)
- Categorical health status (excellent, good, fair, poor, critical)

### 🎯 Anomaly Detection
- Multiple detection methods (Isolation Forest, Elliptic Envelope, Statistical)
- Anomaly scoring and ranking
- Configurable contamination thresholds
- Detailed anomaly summaries

### ⏱️ Remaining Useful Life (RUL) Prediction
- Multiple model types (Random Forest, Gradient Boosting, Linear Regression)
- Confidence interval estimation
- Feature importance analysis
- Degradation rate estimation

### ⚠️ Failure Prediction
- Binary failure classification
- Probability-based risk assessment
- Risk levels (Low, Medium, High, Critical)
- Time-to-failure estimation
- Feature importance for root cause analysis

### 📈 Forecasting
- Time series forecasting for equipment metrics
- Multi-step ahead predictions
- Confidence intervals via bootstrap
- Forecast anomaly detection

### 🚨 Alert Management
- Automated alert generation based on multiple criteria
- Severity levels (INFO, WARNING, CRITICAL, EMERGENCY)
- Alert acknowledgment and tracking
- Comprehensive alert summaries

## Installation

```bash
# Clone the repository
git clone https://github.com/manojsai7/predictive-maintenance.git
cd predictive-maintenance

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Quick Start

```python
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

# Load your sensor data
data = pd.DataFrame({
    'temperature': [70, 72, 71, 73, 75, 78, 82],
    'vibration': [5, 5.1, 5.2, 5.5, 6, 6.5, 7],
    'pressure': [100, 99, 98, 97, 95, 93, 90]
})

# 1. Process the data
processor = DataProcessor()
processed_data = processor.prepare_data(
    data, 
    sensor_columns=['temperature', 'vibration', 'pressure']
)

# 2. Calculate health score
health_scorer = HealthScorer()
health_scorer.set_baseline(data[:3], ['temperature', 'vibration', 'pressure'])
health_score = health_scorer.calculate_overall_health(data.iloc[-1])
print(f"Current Health: {health_score:.1f}/100")

# 3. Detect anomalies
anomaly_detector = AnomalyDetector(method="isolation_forest")
anomaly_detector.fit(data)
anomalies = anomaly_detector.get_anomaly_summary(data)
print(f"Anomalies Detected: {anomalies['anomaly_count']}")

# 4. Predict RUL (requires historical RUL labels)
# rul_predictor = RULPredictor()
# rul_predictor.fit(training_data, rul_column='rul')
# remaining_life = rul_predictor.predict(current_data)

# 5. Generate alerts
alert_manager = AlertManager()
alert_manager.check_health_alerts(health_score, equipment_id="PUMP-001")
active_alerts = alert_manager.get_active_alerts()
print(f"Active Alerts: {len(active_alerts)}")
```

## Complete Example

Run the complete workflow example:

```bash
python examples/complete_workflow.py
```

This example demonstrates:
- Data generation and preprocessing
- Health score calculation
- Anomaly detection
- RUL prediction
- Failure probability estimation
- Forecasting future equipment state
- Alert generation and management
- Comprehensive maintenance report

## Module Documentation

### DataProcessor

Handles data preprocessing and feature engineering:

```python
processor = DataProcessor()

# Prepare data with all preprocessing steps
processed = processor.prepare_data(
    data,
    sensor_columns=['temp', 'vibration'],
    normalize=True,
    remove_outliers=True
)

# Individual operations
clean_data = processor.handle_missing_values(data, strategy='interpolate')
normalized = processor.normalize_features(data)
with_features = processor.extract_features(data, ['temp', 'vibration'])
```

### HealthScorer

Calculates equipment health scores:

```python
scorer = HealthScorer()

# Set baseline from healthy operation period
scorer.set_baseline(healthy_data, ['temp', 'vibration', 'pressure'])

# Set importance weights
scorer.set_weights({'temp': 0.4, 'vibration': 0.3, 'pressure': 0.3})

# Calculate health
health = scorer.calculate_overall_health(current_data)
category = scorer.get_health_category(health)
trend = scorer.get_health_trend(historical_health_scores)
```

### AnomalyDetector

Detects abnormal patterns in equipment data:

```python
detector = AnomalyDetector(method="isolation_forest", contamination=0.1)

# Train on normal operation data
detector.fit(training_data, feature_columns=['temp', 'vibration'])

# Detect anomalies
labels = detector.detect(test_data)
summary = detector.get_anomaly_summary(test_data)
```

### RULPredictor

Predicts remaining useful life:

```python
predictor = RULPredictor(model_type="random_forest")

# Train with historical data
predictor.fit(training_data, rul_column='rul', feature_columns=['temp', 'vibration'])

# Predict RUL with confidence intervals
result = predictor.predict_with_confidence(current_data)
rul = result['predictions']
lower_bound = result['lower_bound']
upper_bound = result['upper_bound']
```

### FailurePredictor

Predicts equipment failure probability:

```python
predictor = FailurePredictor(model_type="random_forest")

# Train with labeled failure data
predictor.fit(training_data, failure_column='failure')

# Get failure predictions with risk levels
result = predictor.predict_with_confidence(test_data)
probabilities = result['probabilities']
risk_levels = result['risk_level']  # 0=Low, 1=Medium, 2=High, 3=Critical
```

### ForecastingEngine

Forecasts future equipment states:

```python
engine = ForecastingEngine(forecast_horizon=20)

# Train forecasting models
engine.fit(historical_data, target_columns=['temperature', 'vibration'])

# Forecast with confidence intervals
forecast = engine.forecast_with_confidence(
    recent_data,
    column='temperature',
    steps=20
)
```

### AlertManager

Manages alerts and warnings:

```python
manager = AlertManager()

# Check various alert conditions
manager.check_health_alerts(health_score, equipment_id="PUMP-001")
manager.check_anomaly_alerts(anomaly_count, total_samples, equipment_id="PUMP-001")
manager.check_failure_alerts(failure_probability, equipment_id="PUMP-001")
manager.check_rul_alerts(remaining_life, equipment_id="PUMP-001")

# Get and manage alerts
active_alerts = manager.get_active_alerts()
summary = manager.get_alert_summary()
manager.acknowledge_alert(alert_id)
```

## Testing

Run the test suite:

```bash
# Install test dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_data_processor.py -v

# Run with coverage
pytest tests/ --cov=predictive_maintenance --cov-report=html
```

## Use Cases

### Manufacturing
- Monitor production equipment health
- Predict machine failures before production stops
- Schedule maintenance during planned downtime
- Reduce unexpected breakdowns

### Energy & Utilities
- Monitor turbine and generator health
- Predict component failures in power plants
- Optimize maintenance schedules
- Extend equipment lifespan

### Transportation
- Monitor vehicle fleet health
- Predict component failures
- Optimize maintenance scheduling
- Reduce vehicle downtime

### Oil & Gas
- Monitor drilling equipment
- Predict pump and compressor failures
- Prevent costly production interruptions
- Ensure safety compliance

## Requirements

- Python >= 3.8
- numpy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- scipy >= 1.7.0

Optional:
- matplotlib >= 3.4.0 (for visualization)
- seaborn >= 0.11.0 (for visualization)
- pytest >= 7.0.0 (for testing)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Author

Predictive Maintenance Team

## Acknowledgments

This toolkit combines best practices from:
- Industrial IoT monitoring
- Time series analysis
- Machine learning for predictive maintenance
- Anomaly detection research
- Reliability engineering

