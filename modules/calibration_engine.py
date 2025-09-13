import numpy as np
import pandas as pd
from scipy import signal
from scipy.optimize import curve_fit

class CalibrationEngine:
    """Engine for sensor calibration and noise filtering"""
    
    def __init__(self):
        self.calibration_params = {
            'temperature': {'offset': 0.0, 'slope': 1.0, 'polynomial': []},
            'pressure': {'offset': 0.0, 'slope': 1.0, 'polynomial': []},
            'vibration': {'offset': 0.0, 'slope': 1.0, 'polynomial': []},
            'humidity': {'offset': 0.0, 'slope': 1.0, 'polynomial': []}
        }
        
        self.filter_params = {
            'type': 'moving_average',
            'window': 5,
            'order': 3  # For Savitzky-Golay filter
        }
        
        self.reference_values = {}  # For calculating calibration offsets
    
    def apply_calibration(self, sensor_data):
        """Apply calibration to sensor data"""
        calibrated_data = sensor_data.copy()
        
        for sensor_type in ['temperature', 'pressure', 'vibration', 'humidity']:
            if sensor_type in sensor_data:
                raw_value = sensor_data[sensor_type]
                
                # Apply linear calibration
                calibrated_value = self._apply_linear_calibration(raw_value, sensor_type)
                
                # Apply polynomial correction if available
                if self.calibration_params[sensor_type]['polynomial']:
                    calibrated_value = self._apply_polynomial_calibration(calibrated_value, sensor_type)
                
                calibrated_data[sensor_type] = calibrated_value
        
        return calibrated_data
    
    def apply_batch_calibration(self, data_df):
        """Apply calibration to a batch of data"""
        calibrated_df = data_df.copy()
        
        for sensor_type in ['temperature', 'pressure', 'vibration', 'humidity']:
            if sensor_type in data_df.columns:
                # Apply calibration
                calibrated_values = data_df[sensor_type].apply(
                    lambda x: self._apply_linear_calibration(x, sensor_type)
                )
                
                # Apply polynomial correction
                if self.calibration_params[sensor_type]['polynomial']:
                    calibrated_values = calibrated_values.apply(
                        lambda x: self._apply_polynomial_calibration(x, sensor_type)
                    )
                
                # Apply noise filtering
                filtered_values = self._apply_noise_filter(calibrated_values.values)
                
                calibrated_df[sensor_type] = filtered_values
        
        return calibrated_df
    
    def _apply_linear_calibration(self, value, sensor_type):
        """Apply linear calibration: y = slope * x + offset"""
        params = self.calibration_params[sensor_type]
        return params['slope'] * value + params['offset']
    
    def _apply_polynomial_calibration(self, value, sensor_type):
        """Apply polynomial calibration"""
        polynomial = self.calibration_params[sensor_type]['polynomial']
        if polynomial:
            return np.polyval(polynomial, value)
        return value
    
    def _apply_noise_filter(self, values):
        """Apply noise filtering to data"""
        if len(values) < self.filter_params['window']:
            return values
        
        filter_type = self.filter_params['type']
        window_size = self.filter_params['window']
        
        if filter_type == 'moving_average':
            return self._moving_average_filter(values, window_size)
        elif filter_type == 'savitzky_golay':
            return self._savitzky_golay_filter(values, window_size)
        elif filter_type == 'butterworth':
            return self._butterworth_filter(values)
        else:
            return values
    
    def _moving_average_filter(self, values, window_size):
        """Apply moving average filter"""
        filtered = np.convolve(values, np.ones(window_size)/window_size, mode='same')
        
        # Handle edge effects
        for i in range(window_size//2):
            filtered[i] = np.mean(values[:i+window_size//2+1])
            filtered[-(i+1)] = np.mean(values[-(i+window_size//2+1):])
        
        return filtered
    
    def _savitzky_golay_filter(self, values, window_size):
        """Apply Savitzky-Golay filter"""
        order = min(self.filter_params['order'], window_size - 1)
        return signal.savgol_filter(values, window_size, order)
    
    def _butterworth_filter(self, values, cutoff_freq=0.1):
        """Apply Butterworth low-pass filter"""
        nyquist = 0.5
        normal_cutoff = cutoff_freq / nyquist
        b, a = signal.butter(4, normal_cutoff, btype='low', analog=False)
        return signal.filtfilt(b, a, values)
    
    def calculate_calibration_offset(self, measured_values, reference_values, sensor_type):
        """Calculate calibration offset from reference measurements"""
        if len(measured_values) != len(reference_values):
            raise ValueError("Measured and reference values must have same length")
        
        measured = np.array(measured_values)
        reference = np.array(reference_values)
        
        # Calculate linear relationship
        slope, intercept = np.polyfit(measured, reference, 1)
        
        # Update calibration parameters
        self.calibration_params[sensor_type]['slope'] = slope
        self.calibration_params[sensor_type]['offset'] = intercept
        
        # Calculate accuracy metrics
        corrected = slope * measured + intercept
        mse = np.mean((corrected - reference) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(corrected - reference))
        
        return {
            'slope': slope,
            'offset': intercept,
            'rmse': rmse,
            'mae': mae,
            'r_squared': np.corrcoef(corrected, reference)[0, 1] ** 2
        }
    
    def fit_polynomial_calibration(self, measured_values, reference_values, sensor_type, degree=2):
        """Fit polynomial calibration curve"""
        measured = np.array(measured_values)
        reference = np.array(reference_values)
        
        # Fit polynomial
        polynomial = np.polyfit(measured, reference, degree)
        self.calibration_params[sensor_type]['polynomial'] = polynomial
        
        # Calculate accuracy
        corrected = np.polyval(polynomial, measured)
        rmse = np.sqrt(np.mean((corrected - reference) ** 2))
        
        return {
            'polynomial': polynomial,
            'rmse': rmse,
            'degree': degree
        }
    
    def auto_calibrate(self, data_df, sensor_type, reference_column=None):
        """Automatic calibration using statistical methods"""
        if reference_column and reference_column in data_df.columns:
            # Use reference column for calibration
            measured = data_df[sensor_type].values
            reference = data_df[reference_column].values
            return self.calculate_calibration_offset(measured, reference, sensor_type)
        else:
            # Use statistical calibration (remove systematic bias)
            values = data_df[sensor_type].values
            
            # Remove outliers (beyond 3 standard deviations)
            mean_val = np.mean(values)
            std_val = np.std(values)
            clean_values = values[np.abs(values - mean_val) <= 3 * std_val]
            
            # Estimate bias and drift
            if len(clean_values) > 10:
                # Simple linear trend removal
                x = np.arange(len(clean_values))
                slope, intercept = np.polyfit(x, clean_values, 1)
                
                # Update calibration to remove trend
                current_offset = self.calibration_params[sensor_type]['offset']
                self.calibration_params[sensor_type]['offset'] = current_offset - intercept
                
                return {
                    'trend_slope': slope,
                    'bias_correction': -intercept,
                    'std_deviation': np.std(clean_values)
                }
    
    def update_parameters(self, calibration_params, filter_params):
        """Update calibration and filter parameters"""
        self.calibration_params.update(calibration_params)
        self.filter_params.update(filter_params)
    
    def get_calibration_summary(self):
        """Get summary of current calibration parameters"""
        summary = {}
        for sensor_type, params in self.calibration_params.items():
            summary[sensor_type] = {
                'linear_offset': params['offset'],
                'linear_slope': params['slope'],
                'has_polynomial': bool(params['polynomial']),
                'polynomial_degree': len(params['polynomial']) - 1 if params['polynomial'] else 0
            }
        
        summary['filter'] = self.filter_params.copy()
        return summary
