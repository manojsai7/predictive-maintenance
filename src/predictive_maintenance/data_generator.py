"""Synthetic data generator for predictive maintenance."""
import numpy as np
import pandas as pd
from typing import Optional, Tuple


class SyntheticDataGenerator:
    """Generate synthetic sensor data for predictive maintenance."""
    
    def __init__(self, random_state: Optional[int] = 42):
        """
        Initialize synthetic data generator.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
    
    def generate_sensor_data(self, 
                            n_units: int = 100,
                            n_cycles_range: Tuple[int, int] = (100, 200),
                            n_sensors: int = 10,
                            failure_probability: float = 0.3) -> pd.DataFrame:
        """
        Generate synthetic sensor data for multiple units.
        
        Args:
            n_units: Number of equipment units
            n_cycles_range: Range of operating cycles (min, max)
            n_sensors: Number of sensors
            failure_probability: Probability that a unit will fail
            
        Returns:
            DataFrame with synthetic sensor data
        """
        data = []
        
        for unit_id in range(1, n_units + 1):
            # Random number of cycles for this unit
            n_cycles = np.random.randint(n_cycles_range[0], n_cycles_range[1])
            
            # Determine if this unit will fail
            will_fail = np.random.random() < failure_probability
            failure_cycle = n_cycles if will_fail else None
            
            # Generate sensor readings for this unit
            unit_data = self._generate_unit_data(
                unit_id=unit_id,
                n_cycles=n_cycles,
                n_sensors=n_sensors,
                failure_cycle=failure_cycle
            )
            
            data.extend(unit_data)
        
        df = pd.DataFrame(data)
        return df
    
    def _generate_unit_data(self, 
                           unit_id: int,
                           n_cycles: int,
                           n_sensors: int,
                           failure_cycle: Optional[int] = None) -> list:
        """
        Generate sensor data for a single unit.
        
        Args:
            unit_id: Unit identifier
            n_cycles: Number of operating cycles
            n_sensors: Number of sensors
            failure_cycle: Cycle at which failure occurs (None if no failure)
            
        Returns:
            List of data dictionaries
        """
        data = []
        
        # Base sensor values (normal operation)
        base_values = np.random.uniform(50, 100, n_sensors)
        
        for cycle in range(1, n_cycles + 1):
            # Calculate degradation factor
            if failure_cycle:
                # Degradation increases as we approach failure
                degradation_factor = (cycle / failure_cycle) ** 2
            else:
                # Minimal degradation for non-failing units
                degradation_factor = cycle / n_cycles * 0.2
            
            # Add noise and degradation
            noise = np.random.normal(0, 2, n_sensors)
            degradation = base_values * degradation_factor * 0.3
            
            sensor_values = base_values + noise + degradation
            
            # Create data entry
            entry = {
                'unit_id': unit_id,
                'cycle': cycle,
                'timestamp': pd.Timestamp('2024-01-01') + pd.Timedelta(hours=cycle)
            }
            
            # Add sensor readings
            for i in range(n_sensors):
                entry[f'sensor_{i+1}'] = sensor_values[i]
            
            # Add operational settings (temperature, pressure, etc.)
            entry['temperature'] = np.random.uniform(20, 40) + degradation_factor * 10
            entry['pressure'] = np.random.uniform(90, 110) - degradation_factor * 5
            entry['vibration'] = np.random.uniform(0.1, 0.5) + degradation_factor * 0.3
            
            data.append(entry)
        
        return data
    
    def add_rul_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add RUL (Remaining Useful Life) labels to the dataset.
        
        Args:
            df: Input dataframe with 'unit_id' and 'cycle' columns
            
        Returns:
            DataFrame with RUL column added
        """
        df_with_rul = df.copy()
        
        # Calculate max cycle for each unit (failure point)
        max_cycles = df.groupby('unit_id')['cycle'].max().to_dict()
        
        # Calculate RUL for each row
        df_with_rul['RUL'] = df_with_rul.apply(
            lambda row: max_cycles[row['unit_id']] - row['cycle'],
            axis=1
        )
        
        return df_with_rul
    
    def create_failure_labels(self, df: pd.DataFrame, 
                             window: int = 30) -> pd.DataFrame:
        """
        Create binary failure labels (1 if failure within window cycles).
        
        Args:
            df: Input dataframe with RUL column
            window: Window size for failure prediction
            
        Returns:
            DataFrame with failure label column
        """
        df_with_label = df.copy()
        
        if 'RUL' not in df.columns:
            df_with_label = self.add_rul_labels(df_with_label)
        
        df_with_label['will_fail'] = (df_with_label['RUL'] <= window).astype(int)
        
        return df_with_label
    
    def split_train_test(self, df: pd.DataFrame, 
                        test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets by units.
        
        Args:
            df: Input dataframe
            test_size: Fraction of units for test set
            
        Returns:
            Tuple of (train_df, test_df)
        """
        units = df['unit_id'].unique()
        np.random.shuffle(units)
        
        n_test = int(len(units) * test_size)
        test_units = units[:n_test]
        train_units = units[n_test:]
        
        train_df = df[df['unit_id'].isin(train_units)].copy()
        test_df = df[df['unit_id'].isin(test_units)].copy()
        
        return train_df, test_df
    
    def save_dataset(self, df: pd.DataFrame, filepath: str):
        """
        Save dataset to CSV file.
        
        Args:
            df: Dataframe to save
            filepath: Path to save the file
        """
        df.to_csv(filepath, index=False)
        print(f"Dataset saved to {filepath}")
    
    def generate_complete_dataset(self, 
                                 output_dir: str = 'data',
                                 n_units: int = 100) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate complete train and test datasets with RUL labels.
        
        Args:
            output_dir: Directory to save datasets
            n_units: Number of units to generate
            
        Returns:
            Tuple of (train_df, test_df)
        """
        # Generate raw sensor data
        print("Generating synthetic sensor data...")
        df = self.generate_sensor_data(n_units=n_units)
        
        # Add RUL labels
        print("Adding RUL labels...")
        df = self.add_rul_labels(df)
        
        # Add failure labels
        print("Adding failure labels...")
        df = self.create_failure_labels(df, window=30)
        
        # Split into train and test
        print("Splitting into train and test sets...")
        train_df, test_df = self.split_train_test(df, test_size=0.2)
        
        print(f"Generated {len(train_df)} training samples and {len(test_df)} test samples")
        
        return train_df, test_df


if __name__ == "__main__":
    # Example usage
    generator = SyntheticDataGenerator(random_state=42)
    train_df, test_df = generator.generate_complete_dataset(n_units=100)
    
    print("\nTrain dataset shape:", train_df.shape)
    print("Test dataset shape:", test_df.shape)
    print("\nSample data:")
    print(train_df.head())
