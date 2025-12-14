# Contributing to Predictive Maintenance

Thank you for your interest in contributing to this project! This guide will help you get started.

## Development Setup

1. **Fork and clone the repository**
```bash
git clone https://github.com/yourusername/predictive-maintenance.git
cd predictive-maintenance
```

2. **Set up your development environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

3. **Install development dependencies**
```bash
pip install pytest pytest-cov black flake8
```

## Code Style

We follow PEP 8 style guidelines with some modifications:
- Line length: 100 characters
- Use `black` for automatic code formatting
- Use `flake8` for linting

Format your code before committing:
```bash
black src/ tests/
flake8 src/ tests/ --max-line-length=100
```

## Testing

All contributions should include tests. We use `pytest` for testing.

### Running Tests

Run all tests:
```bash
pytest tests/ -v
```

Run with coverage:
```bash
pytest tests/ --cov=predictive_maintenance --cov-report=html
```

Run specific test file:
```bash
pytest tests/test_features.py -v
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files as `test_*.py`
- Name test functions as `test_*`
- Use descriptive test names
- Include docstrings explaining what the test does

Example:
```python
def test_feature_extraction():
    """Test that feature extraction works correctly with sample data."""
    # Arrange
    df = create_sample_dataframe()
    
    # Act
    features = extract_features(df)
    
    # Assert
    assert 'sensor_1_mean' in features.columns
    assert len(features) > 0
```

## Adding New Features

1. **Create a new branch**
```bash
git checkout -b feature/your-feature-name
```

2. **Write your code**
   - Follow the existing code structure
   - Add docstrings to all public functions
   - Keep functions focused and small

3. **Add tests**
   - Aim for >80% code coverage
   - Test edge cases

4. **Update documentation**
   - Update README.md if needed
   - Add docstrings
   - Update relevant notebooks

5. **Commit your changes**
```bash
git add .
git commit -m "Add: Brief description of your changes"
```

## Pull Request Process

1. **Update your branch with the latest main**
```bash
git fetch origin
git rebase origin/main
```

2. **Run tests and linting**
```bash
pytest tests/ -v
black src/ tests/
flake8 src/ tests/ --max-line-length=100
```

3. **Push your branch**
```bash
git push origin feature/your-feature-name
```

4. **Create a Pull Request**
   - Provide a clear description of the changes
   - Reference any related issues
   - Include screenshots if applicable
   - Wait for review and address feedback

## Code Review Guidelines

When reviewing code:
- Be respectful and constructive
- Check for code quality and style
- Verify tests are included and passing
- Look for potential bugs or edge cases
- Suggest improvements

## Types of Contributions

We welcome various types of contributions:

### Bug Reports
- Use the issue tracker
- Describe the bug clearly
- Include steps to reproduce
- Provide system information

### Feature Requests
- Open an issue to discuss the feature
- Explain the use case
- Consider implementation details

### Documentation
- Fix typos
- Improve explanations
- Add examples
- Update outdated information

### Code Contributions
- Bug fixes
- New features
- Performance improvements
- Test coverage improvements

## Project Structure

```
predictive-maintenance/
├── src/predictive_maintenance/  # Main package
│   ├── features/               # Feature engineering
│   ├── models/                 # ML models
│   ├── inference/              # API service
│   ├── monitoring/             # Drift detection
│   └── data_quality/           # Data validation
├── tests/                      # Test suite
├── notebooks/                  # Jupyter notebooks
├── configs/                    # Configuration files
└── examples/                   # Example scripts
```

## Commit Message Guidelines

Use clear and descriptive commit messages:

- `Add: New feature or functionality`
- `Fix: Bug fix`
- `Update: Changes to existing functionality`
- `Docs: Documentation changes`
- `Test: Adding or updating tests`
- `Refactor: Code refactoring`
- `Style: Code style changes`

Example:
```
Add: LSTM model for time-series RUL prediction

- Implement LSTMModel class with PyTorch
- Add sequence preparation utilities
- Include model training and prediction methods
- Add tests for LSTM model
```

## Questions?

If you have questions or need help:
- Open an issue for discussion
- Check existing issues and documentation
- Reach out to maintainers

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on what is best for the community
- Show empathy towards others

Thank you for contributing! 🚀
