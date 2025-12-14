"""Tests for classical ML models."""
import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from predictive_maintenance.models.classical_models import RULPredictor, FailureClassifier


def test_rul_predictor_initialization():
    """Test RULPredictor initialization."""
    predictor = RULPredictor(model_type='random_forest')
    assert predictor.model_type == 'random_forest'
    assert predictor.model is None


def test_rul_predictor_build_model():
    """Test model building."""
    predictor = RULPredictor(model_type='random_forest')
    predictor.build_model(n_estimators=10, max_depth=5)
    assert predictor.model is not None


def test_rul_predictor_train_and_predict():
    """Test model training and prediction."""
    # Create sample data
    X_train = pd.DataFrame(np.random.randn(100, 5), columns=[f'feature_{i}' for i in range(5)])
    y_train = np.random.randint(0, 100, 100)
    
    predictor = RULPredictor(model_type='random_forest')
    predictor.build_model(n_estimators=10)
    predictor.train(X_train, y_train)
    
    # Make predictions
    X_test = pd.DataFrame(np.random.randn(20, 5), columns=[f'feature_{i}' for i in range(5)])
    predictions = predictor.predict(X_test)
    
    assert len(predictions) == 20
    assert all(predictions >= 0)  # RUL should be non-negative


def test_rul_predictor_save_load():
    """Test model saving and loading."""
    # Create and train model
    X_train = pd.DataFrame(np.random.randn(100, 5), columns=[f'feature_{i}' for i in range(5)])
    y_train = np.random.randint(0, 100, 100)
    
    predictor = RULPredictor(model_type='random_forest')
    predictor.build_model(n_estimators=10)
    predictor.train(X_train, y_train)
    
    # Save model
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
        model_path = f.name
    
    try:
        predictor.save_model(model_path)
        
        # Load model
        new_predictor = RULPredictor()
        new_predictor.load_model(model_path)
        
        assert new_predictor.model is not None
        assert new_predictor.model_type == 'random_forest'
    finally:
        if os.path.exists(model_path):
            os.remove(model_path)


def test_failure_classifier_initialization():
    """Test FailureClassifier initialization."""
    classifier = FailureClassifier(model_type='random_forest')
    assert classifier.model_type == 'random_forest'
    assert classifier.model is None


def test_failure_classifier_train_and_predict():
    """Test classifier training and prediction."""
    # Create sample data
    X_train = pd.DataFrame(np.random.randn(100, 5), columns=[f'feature_{i}' for i in range(5)])
    y_train = np.random.randint(0, 2, 100)
    
    classifier = FailureClassifier(model_type='random_forest')
    classifier.build_model(n_estimators=10)
    classifier.train(X_train, y_train)
    
    # Make predictions
    X_test = pd.DataFrame(np.random.randn(20, 5), columns=[f'feature_{i}' for i in range(5)])
    predictions = classifier.predict(X_test)
    
    assert len(predictions) == 20
    assert all((predictions == 0) | (predictions == 1))  # Binary classification


def test_failure_classifier_predict_proba():
    """Test probability prediction."""
    # Create sample data
    X_train = pd.DataFrame(np.random.randn(100, 5), columns=[f'feature_{i}' for i in range(5)])
    y_train = np.random.randint(0, 2, 100)
    
    classifier = FailureClassifier(model_type='random_forest')
    classifier.build_model(n_estimators=10)
    classifier.train(X_train, y_train)
    
    # Make probability predictions
    X_test = pd.DataFrame(np.random.randn(20, 5), columns=[f'feature_{i}' for i in range(5)])
    probabilities = classifier.predict_proba(X_test)
    
    assert len(probabilities) == 20
    assert all((probabilities >= 0) & (probabilities <= 1))  # Probabilities between 0 and 1
