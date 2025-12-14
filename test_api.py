"""Test script for FastAPI inference service."""
import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"


def test_health_check():
    """Test health check endpoint."""
    response = requests.get(f"{BASE_URL}/health")
    print(f"Health Check: {response.status_code}")
    print(json.dumps(response.json(), indent=2))


def test_prediction():
    """Test prediction endpoint."""
    sensor_reading = {
        "timestamp": "2024-01-01T12:00:00",
        "sensor_values": {
            "sensor_1": 75.5,
            "sensor_2": 82.3,
            "sensor_3": 68.9,
            "sensor_4": 91.2
        },
        "unit_id": 1,
        "cycle": 100
    }
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=sensor_reading
    )
    
    print(f"\nPrediction: {response.status_code}")
    print(json.dumps(response.json(), indent=2))


def test_batch_prediction():
    """Test batch prediction endpoint."""
    batch_data = {
        "readings": [
            {
                "timestamp": "2024-01-01T12:00:00",
                "sensor_values": {"sensor_1": 75.5, "sensor_2": 82.3},
                "unit_id": 1,
                "cycle": 100
            },
            {
                "timestamp": "2024-01-01T13:00:00",
                "sensor_values": {"sensor_1": 76.2, "sensor_2": 83.1},
                "unit_id": 1,
                "cycle": 101
            }
        ]
    }
    
    response = requests.post(
        f"{BASE_URL}/predict/batch",
        json=batch_data
    )
    
    print(f"\nBatch Prediction: {response.status_code}")
    print(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Predictive Maintenance API")
    print("=" * 60)
    print("\nMake sure the API is running:")
    print("uvicorn predictive_maintenance.inference.api:app --host 0.0.0.0 --port 8000")
    print()
    
    try:
        test_health_check()
        test_prediction()
        test_batch_prediction()
        
        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)
    except requests.exceptions.ConnectionError:
        print("\nERROR: Could not connect to API server.")
        print("Please start the server first:")
        print("uvicorn predictive_maintenance.inference.api:app --host 0.0.0.0 --port 8000")
    except Exception as e:
        print(f"\nERROR: {str(e)}")
