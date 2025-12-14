# Models Directory

This directory stores trained model files.

## Supported Formats
- `.pkl` - Scikit-learn models (via joblib)
- `.pt` / `.pth` - PyTorch models
- `.h5` - Keras/TensorFlow models

## Usage
Models are automatically saved here when using the save_model() methods.

Example:
```python
from predictive_maintenance.models.classical_models import RULPredictor

model = RULPredictor()
# ... train model ...
model.save_model('models/my_rul_model.pkl')
```

## Loading Models
```python
model = RULPredictor()
model.load_model('models/my_rul_model.pkl')
```
