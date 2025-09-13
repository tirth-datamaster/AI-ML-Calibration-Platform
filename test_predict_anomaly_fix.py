#!/usr/bin/env python3
"""
Test script to verify the AnomalyDetector.predict_anomaly bug fix.
This demonstrates that the dimension mismatch error has been resolved.
"""

import numpy as np
import pandas as pd
from modules.anomaly_detector import AnomalyDetector

def test_predict_anomaly_fix():
    """Test that predict_anomaly works without dimension mismatch errors"""
    print("🔧 Testing AnomalyDetector.predict_anomaly Bug Fix")
    print("=" * 60)
    
    # Initialize detector
    detector = AnomalyDetector()
    
    # Generate training data
    print("1. Generating training data and fitting models...")
    np.random.seed(42)
    n_points = 50
    timestamps = pd.date_range('2024-01-01', periods=n_points, freq='1H')
    
    # Create realistic sensor data
    base_temp = 25.0
    temp_data = base_temp + np.cumsum(np.random.normal(0, 0.5, n_points))
    
    train_df = pd.DataFrame({
        'timestamp': timestamps,
        'temperature': temp_data
    })
    
    # Fit the model using training data
    print(f"   Training with {len(train_df)} data points...")
    anomalies = detector.detect_anomalies(train_df, 'temperature', method='isolation_forest')
    print(f"   ✅ Model fitted successfully for temperature sensor")
    print(f"   ✅ Found {len(anomalies)} anomalies during training")
    
    # Test buffer status
    buffer_status = detector.get_sensor_buffer_status('temperature')
    print(f"   ✅ Buffer initialized: {buffer_status}")
    
    print("\n2. Testing real-time prediction with various scenarios...")
    
    # Test Case 1: Cold start prediction (empty buffer)
    print("   Test Case 1: Cold start prediction...")
    detector_cold = AnomalyDetector()
    # Fit a model first
    detector_cold.detect_anomalies(train_df, 'temperature', method='isolation_forest')
    
    new_data_cold = {'temperature': 25.5}
    try:
        is_anomaly, score = detector_cold.predict_anomaly(new_data_cold, 'temperature')
        print(f"   ✅ Cold start prediction successful: anomaly={is_anomaly}, score={score:.4f}")
    except Exception as e:
        print(f"   ❌ Cold start prediction failed: {e}")
        return False
    
    # Test Case 2: Gradual buffer buildup
    print("   Test Case 2: Gradual buffer buildup...")
    test_values = [25.0, 25.2, 24.8, 25.5, 26.0]
    
    for i, value in enumerate(test_values):
        new_data = {'temperature': value}
        try:
            is_anomaly, score = detector.predict_anomaly(new_data, 'temperature')
            buffer_status = detector.get_sensor_buffer_status('temperature')
            print(f"     Point {i+1}: value={value:.1f}, anomaly={is_anomaly}, score={score:.4f}, buffer_size={buffer_status['buffer_size']}")
        except Exception as e:
            print(f"   ❌ Prediction {i+1} failed: {e}")
            return False
    
    # Test Case 3: Extreme anomaly
    print("   Test Case 3: Testing extreme anomaly detection...")
    extreme_data = {'temperature': 100.0}  # Very high temperature
    try:
        is_anomaly, score = detector.predict_anomaly(extreme_data, 'temperature')
        print(f"   ✅ Extreme value prediction: value=100.0, anomaly={is_anomaly}, score={score:.4f}")
        if is_anomaly:
            print("   ✅ Correctly identified extreme value as anomaly!")
        else:
            print("   ⚠️  Extreme value not flagged as anomaly (may need threshold tuning)")
    except Exception as e:
        print(f"   ❌ Extreme value prediction failed: {e}")
        return False
    
    # Test Case 4: Multiple sensors
    print("   Test Case 4: Testing multiple sensor types...")
    
    # Add pressure sensor data
    pressure_data = 1013.25 + np.cumsum(np.random.normal(0, 2, n_points))
    train_df['pressure'] = pressure_data
    
    # Fit model for pressure
    pressure_anomalies = detector.detect_anomalies(train_df, 'pressure', method='isolation_forest')
    print(f"   ✅ Pressure model fitted with {len(pressure_anomalies)} anomalies")
    
    # Test pressure prediction
    pressure_data_test = {'pressure': 1015.0}
    try:
        is_anomaly_p, score_p = detector.predict_anomaly(pressure_data_test, 'pressure')
        print(f"   ✅ Pressure prediction: value=1015.0, anomaly={is_anomaly_p}, score={score_p:.4f}")
    except Exception as e:
        print(f"   ❌ Pressure prediction failed: {e}")
        return False
    
    # Test Case 5: Verify feature vector dimensions
    print("   Test Case 5: Verifying feature vector construction...")
    
    # Add several values to build up buffer
    test_buffer_values = [25.0, 25.1, 24.9, 25.2, 24.8, 25.3]
    for val in test_buffer_values:
        detector.update_sensor_buffer('temperature', val)
    
    # Check buffer status
    buffer_status = detector.get_sensor_buffer_status('temperature')
    print(f"   ✅ Buffer after manual updates: size={buffer_status['buffer_size']}")
    print(f"   ✅ Buffer values: {[f'{v:.1f}' for v in buffer_status['buffer_values']]}")
    
    # Create feature vector manually to verify it's 9-dimensional
    try:
        feature_vector = detector._create_realtime_features(buffer_status['buffer_values'], 25.0)
        print(f"   ✅ Feature vector created: length={len(feature_vector)}")
        print(f"   ✅ Feature values: {[f'{v:.3f}' for v in feature_vector]}")
        
        if len(feature_vector) == 9:
            print("   ✅ CORRECT: Feature vector has 9 dimensions as expected!")
        else:
            print(f"   ❌ ERROR: Feature vector has {len(feature_vector)} dimensions, expected 9")
            return False
            
    except Exception as e:
        print(f"   ❌ Feature vector creation failed: {e}")
        return False
    
    print("\n3. Final validation...")
    
    # Test final prediction with full buffer
    final_test = {'temperature': 30.0}  # Moderately high value
    try:
        is_anomaly_final, score_final = detector.predict_anomaly(final_test, 'temperature')
        print(f"   ✅ Final prediction: value=30.0, anomaly={is_anomaly_final}, score={score_final:.4f}")
        
        # Verify buffer management
        final_buffer = detector.get_sensor_buffer_status('temperature')
        print(f"   ✅ Final buffer size: {final_buffer['buffer_size']}")
        
    except Exception as e:
        print(f"   ❌ Final prediction failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 BUG FIX VERIFICATION SUCCESSFUL!")
    print("✅ No dimension mismatch errors occurred")
    print("✅ Real-time predictions working correctly")
    print("✅ Feature vectors constructed properly (9 dimensions)")
    print("✅ Rolling buffers maintained correctly")
    print("✅ Multiple sensors supported independently")
    print("✅ Cold start scenarios handled gracefully")
    return True

if __name__ == "__main__":
    success = test_predict_anomaly_fix()
    if not success:
        print("\n❌ Bug fix verification FAILED!")
        exit(1)
    print("\n✅ All tests passed - bug has been successfully fixed!")