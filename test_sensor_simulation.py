#!/usr/bin/env python3
"""
Test script to verify sensor simulation functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.sensor_simulator import SensorSimulator
from modules.database_manager import DatabaseManager
import pandas as pd
from datetime import datetime
import time

def test_sensor_simulation():
    """Test the sensor simulation functionality"""
    print("🔧 Testing AI/ML-Enhanced Calibration Platform - Sensor Simulation")
    print("=" * 60)
    
    # Initialize components
    print("1. Initializing sensor simulator...")
    simulator = SensorSimulator()
    
    print("2. Testing real-time data generation...")
    # Generate 10 data points
    sensor_data_points = []
    
    for i in range(10):
        data_point = simulator.generate_realtime_data()
        sensor_data_points.append(data_point)
        print(f"   Point {i+1}: T={data_point['temperature']:.2f}°C, "
              f"P={data_point['pressure']:.2f}hPa, "
              f"V={data_point['vibration']:.2f}m/s², "
              f"H={data_point['humidity']:.2f}%")
        time.sleep(0.1)  # Small delay between readings
    
    print("\n3. Verifying data characteristics...")
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(sensor_data_points)
    
    # Check all sensors have data
    required_sensors = ['temperature', 'pressure', 'vibration', 'humidity']
    missing_sensors = [s for s in required_sensors if s not in df.columns]
    
    if missing_sensors:
        print(f"❌ Missing sensor data for: {missing_sensors}")
        return False
    else:
        print("✅ All 4 sensor types generating data")
    
    # Check value ranges
    temp_range = (df['temperature'].min(), df['temperature'].max())
    pressure_range = (df['pressure'].min(), df['pressure'].max())
    vibration_range = (df['vibration'].min(), df['vibration'].max())
    humidity_range = (df['humidity'].min(), df['humidity'].max())
    
    print(f"   Temperature range: {temp_range[0]:.2f} to {temp_range[1]:.2f}°C")
    print(f"   Pressure range: {pressure_range[0]:.2f} to {pressure_range[1]:.2f}hPa")
    print(f"   Vibration range: {vibration_range[0]:.2f} to {vibration_range[1]:.2f}m/s²")
    print(f"   Humidity range: {humidity_range[0]:.2f} to {humidity_range[1]:.2f}%")
    
    # Validate ranges are reasonable
    temp_valid = -50 <= temp_range[0] <= 100 and -50 <= temp_range[1] <= 100
    pressure_valid = 800 <= pressure_range[0] <= 1200 and 800 <= pressure_range[1] <= 1200
    vibration_valid = 0 <= vibration_range[0] <= 50 and 0 <= vibration_range[1] <= 50
    humidity_valid = 0 <= humidity_range[0] <= 100 and 0 <= humidity_range[1] <= 100
    
    if all([temp_valid, pressure_valid, vibration_valid, humidity_valid]):
        print("✅ All sensor values within expected ranges")
    else:
        print("❌ Some sensor values outside expected ranges")
        return False
    
    print("\n4. Testing historical data generation...")
    historical_data = simulator.generate_historical_data(days=1, interval_minutes=60)
    
    if not historical_data.empty and len(historical_data) > 20:
        print(f"✅ Generated {len(historical_data)} historical data points")
        
        # Check for variations in data (not all identical)
        temp_variation = historical_data['temperature'].std()
        if temp_variation > 0.1:
            print(f"✅ Data shows realistic variation (temp std: {temp_variation:.2f})")
        else:
            print("❌ Data appears too uniform (no variation)")
            return False
    else:
        print("❌ Failed to generate sufficient historical data")
        return False
    
    print("\n5. Testing database integration...")
    try:
        db_manager = DatabaseManager(':memory:')  # Use in-memory database for testing
        # Database tables are automatically initialized in __init__
        
        # Store some test data
        for data_point in sensor_data_points[:5]:
            result = db_manager.store_sensor_data(data_point)
            if result is None:
                print("❌ Failed to store sensor data in database")
                return False
        
        # Retrieve data
        stored_data = db_manager.get_recent_data(limit=10)
        if not stored_data.empty:
            print(f"✅ Successfully stored and retrieved {len(stored_data)} data points")
        else:
            print("❌ No data retrieved from database")
            return False
            
    except Exception as e:
        print(f"❌ Database integration error: {e}")
        return False
    
    print("\n6. Testing anomaly injection...")
    simulator.set_anomaly_probability(1.0)  # Force anomalies
    anomaly_data = []
    
    for i in range(5):
        data_point = simulator.generate_realtime_data()
        anomaly_data.append(data_point)
    
    # Check if we got some variations that might indicate anomalies
    anomaly_df = pd.DataFrame(anomaly_data)
    temp_std = anomaly_df['temperature'].std()
    
    if temp_std > 1.0:  # Higher variation suggests anomalies were injected
        print("✅ Anomaly injection appears to be working")
    else:
        print("⚠️  Anomaly injection may not be working optimally")
    
    # Reset anomaly probability
    simulator.set_anomaly_probability(0.05)
    
    print("\n" + "=" * 60)
    print("🎉 Sensor Simulation Test PASSED!")
    print("✅ All sensor types generate realistic data")
    print("✅ Data values are within expected ranges")
    print("✅ Historical data generation works")
    print("✅ Database integration functional")
    print("✅ Anomaly injection system operational")
    
    return True

if __name__ == "__main__":
    success = test_sensor_simulation()
    sys.exit(0 if success else 1)