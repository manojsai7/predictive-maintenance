.PHONY: help install test lint format clean docker-build docker-up docker-down example

help:
	@echo "Predictive Maintenance - Available Commands"
	@echo "==========================================="
	@echo "install       - Install package and dependencies"
	@echo "test          - Run tests"
	@echo "test-cov      - Run tests with coverage"
	@echo "lint          - Run linting checks"
	@echo "format        - Format code with black"
	@echo "clean         - Remove generated files"
	@echo "docker-build  - Build Docker images"
	@echo "docker-up     - Start Docker services"
	@echo "docker-down   - Stop Docker services"
	@echo "example       - Run example workflow"
	@echo "api           - Start FastAPI server"

install:
	pip install -e .
	pip install -r requirements.txt

test:
	PYTHONPATH=src pytest tests/ -v

test-cov:
	PYTHONPATH=src pytest tests/ --cov=predictive_maintenance --cov-report=html --cov-report=term

lint:
	flake8 src/ tests/ --max-line-length=100 --exclude=__pycache__

format:
	black src/ tests/ --line-length=100

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf htmlcov/ .coverage build/ dist/ *.egg-info

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

example:
	python example.py

api:
	PYTHONPATH=src uvicorn predictive_maintenance.inference.api:app --host 0.0.0.0 --port 8000 --reload
