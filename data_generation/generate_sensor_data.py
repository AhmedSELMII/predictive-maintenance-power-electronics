"""
Synthetic sensor data generator for power electronics (IGBT modules)
Based on real-world thermal and electrical characteristics
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

def generate_healthy_operation(n_samples=1000, component_id="IGBT_001"):
    """
    Generate sensor data for healthy IGBT operation
    """
    
    # Time series
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [start_time + timedelta(hours=i) for i in range(n_samples)]
    
    # Operating parameters (healthy ranges)
    load_current = np.random.normal(50, 5, n_samples)  # Amperes, mean=50A
    switching_freq = np.random.normal(10000, 500, n_samples)  # Hz, mean=10kHz
    
    # Thermal parameters
    # Tj = Tc + (Rth * Power_loss)
    ambient_temp = 25 + 10 * np.sin(np.linspace(0, 4*np.pi, n_samples))  # Daily variation
    
    # Thermal resistance (stable for healthy component)
    thermal_resistance = np.random.normal(0.5, 0.02, n_samples)  # K/W
    
    # Power loss (simplified model)
    power_loss = load_current * 1.5 + switching_freq * 0.001
    
    # Case temperature
    case_temp = ambient_temp + np.random.normal(30, 3, n_samples)
    
    # Junction temperature
    junction_temp = case_temp + (thermal_resistance * power_loss)
    
    # Voltage (stable)
    voltage = np.random.normal(600, 10, n_samples)  # Volts
    
    # Operating hours
    operating_hours = np.arange(n_samples)
    
    # Health status
    health_status = np.ones(n_samples)  # 1 = healthy
    
    # Create DataFrame
    data = pd.DataFrame({
        'timestamp': timestamps,
        'component_id': component_id,
        'load_current_A': load_current,
        'switching_frequency_Hz': switching_freq,
        'junction_temp_C': junction_temp,
        'case_temp_C': case_temp,
        'voltage_V': voltage,
        'thermal_resistance_KW': thermal_resistance,
        'ambient_temp_C': ambient_temp,
        'operating_hours': operating_hours,
        'health_status': health_status,
        'failure_occurred': 0
    })
    
    return data


def generate_degrading_operation(n_samples=500, component_id="IGBT_002"):
    """
    Generate sensor data showing gradual degradation leading to failure
    Simulates: thermal resistance increase (solder fatigue, bond wire degradation)
    """
    
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [start_time + timedelta(hours=i) for i in range(n_samples)]
    
    # Operating parameters
    load_current = np.random.normal(55, 6, n_samples)  # Slightly higher stress
    switching_freq = np.random.normal(10000, 500, n_samples)
    
    ambient_temp = 25 + 10 * np.sin(np.linspace(0, 4*np.pi, n_samples))
    
    # DEGRADATION: Thermal resistance increases over time
    degradation_factor = np.linspace(1.0, 2.5, n_samples)  # 2.5x increase
    thermal_resistance = 0.5 * degradation_factor + np.random.normal(0, 0.03, n_samples)
    
    power_loss = load_current * 1.5 + switching_freq * 0.001
    case_temp = ambient_temp + np.random.normal(32, 4, n_samples)
    
    # Junction temp increases as thermal resistance degrades
    junction_temp = case_temp + (thermal_resistance * power_loss)
    
    voltage = np.random.normal(600, 15, n_samples)  # Slightly more variation
    operating_hours = np.arange(n_samples)
    
    # Health degradation
    health_status = np.linspace(1.0, 0.1, n_samples)
    
    # Failure flag (last 10% of samples)
    failure_occurred = np.zeros(n_samples)
    failure_occurred[int(0.9 * n_samples):] = 1
    
    data = pd.DataFrame({
        'timestamp': timestamps,
        'component_id': component_id,
        'load_current_A': load_current,
        'switching_frequency_Hz': switching_freq,
        'junction_temp_C': junction_temp,
        'case_temp_C': case_temp,
        'voltage_V': voltage,
        'thermal_resistance_KW': thermal_resistance,
        'ambient_temp_C': ambient_temp,
        'operating_hours': operating_hours,
        'health_status': health_status,
        'failure_occurred': failure_occurred
    })
    
    return data


def generate_complete_dataset(n_healthy_components=8, n_degrading_components=2):
    """
    Generate complete dataset with multiple components
    """
    all_data = []
    
    # Generate healthy components
    for i in range(n_healthy_components):
        component_id = f"IGBT_{str(i+1).zfill(3)}"
        n_samples = np.random.randint(800, 1200)  # Variable operation time
        data = generate_healthy_operation(n_samples, component_id)
        all_data.append(data)
        print(f"Generated {n_samples} samples for healthy component {component_id}")
    
    # Generate degrading components
    for i in range(n_degrading_components):
        component_id = f"IGBT_{str(n_healthy_components + i + 1).zfill(3)}"
        n_samples = np.random.randint(400, 600)
        data = generate_degrading_operation(n_samples, component_id)
        all_data.append(data)
        print(f"Generated {n_samples} samples for degrading component {component_id}")
    
    # Combine all data
    complete_data = pd.concat(all_data, ignore_index=True)
    
    # Shuffle
    complete_data = complete_data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return complete_data


if __name__ == "__main__":
    print("🔧 Generating synthetic power electronics sensor data...")
    print("=" * 60)
    
    # Generate dataset
    data = generate_complete_dataset(n_healthy_components=8, n_degrading_components=2)
    
    # Save to CSV
    output_file = "data/power_electronics_sensor_data.csv"
    data.to_csv(output_file, index=False)
    
    print("\n✅ Data generation complete!")
    print(f"📊 Total samples: {len(data)}")
    print(f"📁 Saved to: {output_file}")
    print(f"🔍 Failure rate: {data['failure_occurred'].mean()*100:.2f}%")
    print("\n📈 Dataset summary:")
    print(data.describe())
    print("\n🏷️ Component breakdown:")
    print(data.groupby('component_id')['failure_occurred'].agg(['count', 'sum']))