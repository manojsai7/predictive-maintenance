# Data Directory

This directory contains the datasets for predictive maintenance.

## Structure
- `raw/` - Raw sensor data
- `processed/` - Processed and feature-engineered data

## Usage
Place your raw sensor data CSV files in the `raw/` directory.
Processed datasets will be saved to the `processed/` directory.

## Generating Synthetic Data
You can generate synthetic data using the provided data generator:

```python
from predictive_maintenance.data_generator import SyntheticDataGenerator

generator = SyntheticDataGenerator(random_state=42)
train_df, test_df = generator.generate_complete_dataset(n_units=100)
generator.save_dataset(train_df, 'data/train_data.csv')
generator.save_dataset(test_df, 'data/test_data.csv')
```
