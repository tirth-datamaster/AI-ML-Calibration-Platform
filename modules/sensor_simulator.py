import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import math

class SensorSimulator:
    """Simulates realistic sensor data with noise and anomalies"""
    
    def __init__(self):
        self.base_values = {
            'temperature': 25.0,  # °C
            'pressure': 1013.25,  # hPa
            'vibration': 2.0,     # m/s²
            'humidity': 50.0      # %
        }
        
        self.noise_levels = {
            'temperature': 0.5,
            'pressure': 2.0,
            'vibration': 0.3,
            'humidity': 2.0
        }
        
        self.drift_rates = {
            'temperature': 0.001,   # °C per hour
            'pressure': 0.01,       # hPa per hour
            'vibration': 0.0005,    # m/s² per hour
            'humidity': 0.002       # % per hour
        }
        
        self.anomaly_probability = 0.05  # 5% chance of anomaly
        self.start_time = datetime.now()
        self.last_values = self.base_values.copy()
    
    def generate_realtime_data(self):
        """Generate a single data point with realistic patterns"""
        current_time = datetime.now()
        hours_elapsed = (current_time - self.start_time).total_seconds() / 3600
        
        data = {'timestamp': current_time}
        
        for sensor_type in self.base_values.keys():
            # Base value with time-based patterns
            base = self.base_values[sensor_type]
            
            # Add daily pattern (sinusoidal)
            daily_pattern = self._get_daily_pattern(sensor_type, current_time)
            
            # Add gradual drift
            drift = self.drift_rates[sensor_type] * hours_elapsed
            
            # Add noise
            noise = np.random.normal(0, self.noise_levels[sensor_type])
            
            # Calculate value
            value = base + daily_pattern + drift + noise
            
            # Add random anomalies
            if random.random() < self.anomaly_probability:
                value = self._inject_anomaly(value, sensor_type)
            
            # Apply sensor-specific constraints
            value = self._apply_constraints(value, sensor_type)
            
            data[sensor_type] = value
            self.last_values[sensor_type] = value
        
        return data
    
    def generate_historical_data(self, days=30, interval_minutes=5):
        """Generate historical data for analysis"""
        start_time = datetime.now() - timedelta(days=days)
        end_time = datetime.now()
        
        timestamps = pd.date_range(start_time, end_time, freq=f'{interval_minutes}T')
        data = []
        
        for timestamp in timestamps:
            hours_elapsed = (timestamp - start_time).total_seconds() / 3600
            
            row = {'timestamp': timestamp}
            
            for sensor_type in self.base_values.keys():
                base = self.base_values[sensor_type]
                daily_pattern = self._get_daily_pattern(sensor_type, timestamp)
                drift = self.drift_rates[sensor_type] * hours_elapsed
                noise = np.random.normal(0, self.noise_levels[sensor_type])
                
                value = base + daily_pattern + drift + noise
                
                # Random anomalies
                if random.random() < self.anomaly_probability:
                    value = self._inject_anomaly(value, sensor_type)
                
                value = self._apply_constraints(value, sensor_type)
                row[sensor_type] = value
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def _get_daily_pattern(self, sensor_type, timestamp):
        """Generate daily pattern based on sensor type"""
        hour = timestamp.hour
        
        if sensor_type == 'temperature':
            # Temperature peaks in afternoon
            return 5 * math.sin(2 * math.pi * (hour - 6) / 24)
        elif sensor_type == 'pressure':
            # Pressure has subtle daily variation
            return 2 * math.sin(2 * math.pi * hour / 24)
        elif sensor_type == 'vibration':
            # Higher vibration during work hours
            if 8 <= hour <= 18:
                return 1.0 + 0.5 * math.sin(2 * math.pi * (hour - 8) / 10)
            else:
                return 0.2
        elif sensor_type == 'humidity':
            # Humidity inversely related to temperature
            return -3 * math.sin(2 * math.pi * (hour - 6) / 24)
        
        return 0
    
    def _inject_anomaly(self, value, sensor_type):
        """Inject different types of anomalies"""
        anomaly_type = random.choice(['spike', 'dip', 'stuck', 'drift'])
        
        if anomaly_type == 'spike':
            return value + 3 * self.noise_levels[sensor_type] * random.uniform(3, 8)
        elif anomaly_type == 'dip':
            return value - 3 * self.noise_levels[sensor_type] * random.uniform(3, 8)
        elif anomaly_type == 'stuck':
            # Return previous value (sensor stuck)
            return self.last_values.get(sensor_type, value)
        elif anomaly_type == 'drift':
            # Sudden drift
            return value + self.noise_levels[sensor_type] * random.uniform(-10, 10)
        
        return value
    
    def _apply_constraints(self, value, sensor_type):
        """Apply realistic constraints to sensor values"""
        if sensor_type == 'temperature':
            return max(-50, min(100, value))  # -50°C to 100°C
        elif sensor_type == 'pressure':
            return max(800, min(1200, value))  # 800 to 1200 hPa
        elif sensor_type == 'vibration':
            return max(0, min(50, value))  # 0 to 50 m/s²
        elif sensor_type == 'humidity':
            return max(0, min(100, value))  # 0% to 100%
        
        return value
    
    def set_anomaly_probability(self, probability):
        """Set the probability of anomalies"""
        self.anomaly_probability = max(0, min(1, probability))
    
    def reset_simulation(self):
        """Reset simulation to initial state"""
        self.start_time = datetime.now()
        self.last_values = self.base_values.copy()
