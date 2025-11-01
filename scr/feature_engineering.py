"""
Feature engineering for predictive maintenance
Transform raw sensor data into ML-ready features
"""

import pandas as pd
import numpy as np

class FeatureEngineer:
    """
    Create time-series features for failure prediction
    """
    
    def __init__(self, window_size=24):
        """
        Args:
            window_size: Number of hours for rolling window calculations
        """
        self.window_size = window_size
    
    def create_rolling_features(self, df, columns):
        """
        Create rolling statistics (mean, std, min, max, trend)
        """
        result_df = df.copy()
        
        for col in columns:
            # Rolling mean
            result_df[f'{col}_rolling_mean'] = df.groupby('component_id')[col].transform(
                lambda x: x.rolling(window=self.window_size, min_periods=1).mean()
            )
            
            # Rolling std (volatility)
            result_df[f'{col}_rolling_std'] = df.groupby('component_id')[col].transform(
                lambda x: x.rolling(window=self.window_size, min_periods=1).std()
            )
            
            # Rolling min/max
            result_df[f'{col}_rolling_min'] = df.groupby('component_id')[col].transform(
                lambda x: x.rolling(window=self.window_size, min_periods=1).min()
            )
            
            result_df[f'{col}_rolling_max'] = df.groupby('component_id')[col].transform(
                lambda x: x.rolling(window=self.window_size, min_periods=1).max()
            )
            
            # Trend (difference from rolling mean)
            result_df[f'{col}_trend'] = df[col] - result_df[f'{col}_rolling_mean']
            
        return result_df
    
    def create_rate_of_change(self, df, columns):
        """
        Calculate rate of change (delta) for key parameters
        """
        result_df = df.copy()
        
        for col in columns:
            # First-order difference (hour-to-hour change)
            result_df[f'{col}_delta'] = df.groupby('component_id')[col].diff()
            
            # Rate of change over window
            result_df[f'{col}_rate_of_change'] = df.groupby('component_id')[col].transform(
                lambda x: x.pct_change(periods=self.window_size)
            )
        
        return result_df
    
    def create_thermal_features(self, df):
        """
        Domain-specific features based on power electronics knowledge
        """
        result_df = df.copy()
        
        # Temperature difference (Tj - Tc) - key indicator of thermal resistance
        result_df['temp_difference'] = df['junction_temp_C'] - df['case_temp_C']
        
        # Thermal stress index (normalized)
        result_df['thermal_stress_index'] = (
            df['junction_temp_C'] / 150  # 150°C is typical max rating
        ) * (df['load_current_A'] / 100)  # normalized current
        
        # Power dissipation estimate
        result_df['estimated_power_loss'] = (
            df['load_current_A'] * 1.5 + df['switching_frequency_Hz'] * 0.001
        )
        
        # Thermal cycling indicator (std of junction temp)
        result_df['thermal_cycling'] = df.groupby('component_id')['junction_temp_C'].transform(
            lambda x: x.rolling(window=self.window_size, min_periods=1).std()
        )
        
        return result_df
    
    def create_cumulative_features(self, df):
        """
        Cumulative stress indicators
        """
        result_df = df.copy()
        
        # Cumulative thermal stress (area under curve)
        result_df['cumulative_thermal_stress'] = df.groupby('component_id')['junction_temp_C'].cumsum()
        
        # Cumulative high-temp exposure (hours above 100°C)
        result_df['high_temp_exposure'] = df.groupby('component_id')['junction_temp_C'].transform(
            lambda x: (x > 100).cumsum()
        )
        
        # Cumulative switching cycles
        result_df['cumulative_switches'] = df.groupby('component_id')['switching_frequency_Hz'].cumsum()
        
        return result_df
    
    def create_all_features(self, df):
        """
        Create complete feature set
        """
        print("🔧 Starting feature engineering...")
        
        # Convert timestamp to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by component and time
        df = df.sort_values(['component_id', 'timestamp']).reset_index(drop=True)
        
        # Key columns for feature engineering
        sensor_columns = [
            'junction_temp_C',
            'case_temp_C', 
            'thermal_resistance_KW',
            'load_current_A',
            'voltage_V'
        ]
        
        print("  → Creating rolling features...")
        df = self.create_rolling_features(df, sensor_columns)
        
        print("  → Creating rate of change features...")
        df = self.create_rate_of_change(df, sensor_columns)
        
        print("  → Creating thermal features...")
        df = self.create_thermal_features(df)
        
        print("  → Creating cumulative features...")
        df = self.create_cumulative_features(df)
        
        # Fill NaN values (from rolling windows at start)
        df = df.fillna(method='bfill').fillna(0)
        
        print(f"✅ Feature engineering complete! Created {df.shape[1]} features")
        
        return df


def prepare_ml_dataset(input_file, output_file, window_size=24):
    """
    Complete pipeline: load data → engineer features → save
    """
    print("=" * 70)
    print("FEATURE ENGINEERING PIPELINE")
    print("=" * 70)
    
    # Load data
    print(f"\n📂 Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"   Original shape: {df.shape}")
    
    # Engineer features
    engineer = FeatureEngineer(window_size=window_size)
    df_features = engineer.create_all_features(df)
    
    # Save
    print(f"\n💾 Saving engineered dataset to {output_file}...")
    df_features.to_csv(output_file, index=False)
    
    # Summary
    print(f"\n📊 Final dataset shape: {df_features.shape}")
    print(f"   Features created: {df_features.shape[1] - df.shape[1]}")
    
    print("\n🎯 Top features for ML:")
    feature_cols = [col for col in df_features.columns 
                   if col not in ['timestamp', 'component_id', 'health_status', 'failure_occurred']]
    print(f"   Total ML features: {len(feature_cols)}")
    
    return df_features


if __name__ == "__main__":
    # Run feature engineering
    df_features = prepare_ml_dataset(
        input_file='data/power_electronics_sensor_data.csv',
        output_file='data/power_electronics_features.csv',
        window_size=24  # 24-hour rolling window
    )
    
    # Display sample
    print("\n📋 Sample of engineered features:")
    print(df_features.head())
    
    print("\n" + "=" * 70)
    print("✅ FEATURE ENGINEERING COMPLETE!")
    print("=" * 70)