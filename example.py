"""Example script demonstrating predictive maintenance workflow."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from predictive_maintenance.data_generator import SyntheticDataGenerator
from predictive_maintenance.features.feature_extraction import RULFeatureEngineer
from predictive_maintenance.models.classical_models import RULPredictor
from predictive_maintenance.data_quality.checks import DataQualityChecker
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def main():
    """Run complete predictive maintenance workflow."""
    print("=" * 60)
    print("Predictive Maintenance - Example Workflow")
    print("=" * 60)
    
    # Step 1: Generate synthetic data
    print("\n1. Generating synthetic data...")
    generator = SyntheticDataGenerator(random_state=42)
    train_df, test_df = generator.generate_complete_dataset(n_units=50)
    print(f"   Train set: {train_df.shape[0]} samples")
    print(f"   Test set: {test_df.shape[0]} samples")
    
    # Step 2: Data quality checks
    print("\n2. Running data quality checks...")
    checker = DataQualityChecker()
    report = checker.generate_quality_report(train_df)
    print(f"   Quality Score: {report.quality_score:.2f}/100")
    if report.issues:
        print(f"   Issues found: {len(report.issues)}")
        for issue in report.issues[:3]:
            print(f"   - {issue}")
    else:
        print("   No issues found!")
    
    # Step 3: Feature engineering
    print("\n3. Engineering features...")
    sensor_cols = [col for col in train_df.columns if col.startswith('sensor_')]
    feature_engineer = RULFeatureEngineer()
    
    train_features = feature_engineer.create_degradation_features(train_df, sensor_cols)
    test_features = feature_engineer.create_degradation_features(test_df, sensor_cols)
    print(f"   Created {len(train_features.columns)} features")
    
    # Step 4: Prepare training data
    print("\n4. Preparing training data...")
    exclude_cols = ['unit_id', 'cycle', 'timestamp', 'RUL', 'will_fail']
    feature_cols = [col for col in train_features.columns if col not in exclude_cols]
    
    X_train = train_features[feature_cols].fillna(0)
    y_train = train_features['RUL']
    X_test = test_features[feature_cols].fillna(0)
    y_test = test_features['RUL']
    
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Features: {X_train.shape[1]}")
    
    # Step 5: Train model
    print("\n5. Training Random Forest model...")
    model = RULPredictor(model_type='random_forest')
    model.build_model(n_estimators=50, max_depth=10, random_state=42)
    model.train(X_train, y_train)
    print("   Model training complete!")
    
    # Step 6: Evaluate model
    print("\n6. Evaluating model performance...")
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"   RMSE: {rmse:.2f} cycles")
    print(f"   MAE: {mae:.2f} cycles")
    print(f"   R²: {r2:.4f}")
    
    # Step 7: Feature importance
    print("\n7. Top 10 most important features:")
    importance = model.get_feature_importance()
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    for i, (feature, score) in enumerate(sorted_features, 1):
        print(f"   {i}. {feature}: {score:.4f}")
    
    # Step 8: Save model
    print("\n8. Saving model...")
    os.makedirs('models', exist_ok=True)
    model_path = 'models/example_rul_predictor.pkl'
    model.save_model(model_path)
    print(f"   Model saved to: {model_path}")
    
    # Step 9: Make sample predictions
    print("\n9. Making sample predictions...")
    sample_data = X_test.iloc[:5]
    sample_predictions = model.predict(sample_data)
    sample_actual = y_test.iloc[:5].values
    
    print("\n   Sample Predictions vs Actual:")
    print("   " + "-" * 40)
    for i, (pred, actual) in enumerate(zip(sample_predictions, sample_actual), 1):
        error = abs(pred - actual)
        print(f"   Sample {i}: Predicted={pred:.1f}, Actual={actual:.1f}, Error={error:.1f}")
    
    print("\n" + "=" * 60)
    print("Workflow complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
