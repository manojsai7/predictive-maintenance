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
