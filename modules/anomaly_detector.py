import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class AnomalyDetector:
    """Detects anomalies in sensor data using multiple methods"""
    
    def __init__(self):
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        self.scaler = StandardScaler()
        self.fitted_models = {}
        
        # Statistical thresholds
        self.z_score_threshold = 3.0
        self.iqr_multiplier = 1.5
        
        # Pattern detection parameters
        self.stuck_threshold = 0.001  # Threshold for detecting stuck sensors
        self.spike_threshold = 5.0    # Standard deviations for spike detection
    
    def detect_anomalies(self, data_df, sensor_type, method='combined'):
        """Detect anomalies using specified method"""
        if sensor_type not in data_df.columns:
            return pd.DataFrame()
        
        values = data_df[sensor_type].values
        timestamps = data_df['timestamp'].values
        
        anomalies = pd.DataFrame()
        
        if method == 'statistical' or method == 'combined':
            statistical_anomalies = self._detect_statistical_anomalies(
                data_df, sensor_type, timestamps, values
            )
            anomalies = pd.concat([anomalies, statistical_anomalies], ignore_index=True)
        
        if method == 'isolation_forest' or method == 'combined':
            ml_anomalies = self._detect_ml_anomalies(
                data_df, sensor_type, timestamps, values
            )
            anomalies = pd.concat([anomalies, ml_anomalies], ignore_index=True)
        
        if method == 'pattern' or method == 'combined':
            pattern_anomalies = self._detect_pattern_anomalies(
                data_df, sensor_type, timestamps, values
            )
            anomalies = pd.concat([anomalies, pattern_anomalies], ignore_index=True)
        
        # Remove duplicates and sort by timestamp
        if not anomalies.empty:
            anomalies = anomalies.drop_duplicates(subset=['timestamp'])
            anomalies = anomalies.sort_values('timestamp').reset_index(drop=True)
        
        return anomalies
    
    def _detect_statistical_anomalies(self, data_df, sensor_type, timestamps, values):
        """Detect anomalies using statistical methods"""
        anomalies = []
        
        # Z-score based detection
        z_scores = np.abs(stats.zscore(values))
        z_anomalies = np.where(z_scores > self.z_score_threshold)[0]
        
        for idx in z_anomalies:
            anomalies.append({
                'timestamp': timestamps[idx],
                sensor_type: values[idx],
                'anomaly_type': 'z_score_outlier',
                'anomaly_score': z_scores[idx],
                'method': 'statistical'
            })
        
        # IQR based detection
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        lower_bound = q1 - self.iqr_multiplier * iqr
        upper_bound = q3 + self.iqr_multiplier * iqr
        
        iqr_anomalies = np.where((values < lower_bound) | (values > upper_bound))[0]
        
        for idx in iqr_anomalies:
            if idx not in z_anomalies:  # Avoid duplicates
                distance_from_bound = min(
                    abs(values[idx] - lower_bound),
                    abs(values[idx] - upper_bound)
                )
                anomalies.append({
                    'timestamp': timestamps[idx],
                    sensor_type: values[idx],
                    'anomaly_type': 'iqr_outlier',
                    'anomaly_score': distance_from_bound / iqr,
                    'method': 'statistical'
                })
        
        return pd.DataFrame(anomalies)
    
    def _detect_ml_anomalies(self, data_df, sensor_type, timestamps, values):
        """Detect anomalies using machine learning methods"""
        anomalies = []
        
        if len(values) < 10:
            return pd.DataFrame(anomalies)
        
        # Prepare features
        features = self._create_features(data_df, sensor_type)
        
        # Fit and predict using Isolation Forest
        try:
            scaled_features = self.scaler.fit_transform(features)
            predictions = self.isolation_forest.fit_predict(scaled_features)
            anomaly_scores = self.isolation_forest.decision_function(scaled_features)
            
            # Store fitted model
            self.fitted_models[sensor_type] = {
                'isolation_forest': self.isolation_forest,
                'scaler': self.scaler
            }
            
            # Extract anomalies
            anomaly_indices = np.where(predictions == -1)[0]
            
            for idx in anomaly_indices:
                anomalies.append({
                    'timestamp': timestamps[idx],
                    sensor_type: values[idx],
                    'anomaly_type': 'isolation_forest',
                    'anomaly_score': abs(anomaly_scores[idx]),
                    'method': 'machine_learning'
                })
        
        except Exception as e:
            print(f"Error in ML anomaly detection: {e}")
        
        return pd.DataFrame(anomalies)
    
    def _detect_pattern_anomalies(self, data_df, sensor_type, timestamps, values):
        """Detect pattern-based anomalies"""
        anomalies = []
        
        if len(values) < 5:
            return pd.DataFrame(anomalies)
        
        # Detect stuck sensor (consecutive identical values)
        stuck_anomalies = self._detect_stuck_sensor(timestamps, values, sensor_type)
        anomalies.extend(stuck_anomalies)
        
        # Detect spikes and dips
        spike_anomalies = self._detect_spikes_and_dips(timestamps, values, sensor_type)
        anomalies.extend(spike_anomalies)
        
        # Detect trend deviations
        trend_anomalies = self._detect_trend_deviations(timestamps, values, sensor_type)
        anomalies.extend(trend_anomalies)
        
        return pd.DataFrame(anomalies)
    
    def _detect_stuck_sensor(self, timestamps, values, sensor_type):
        """Detect when sensor is stuck (consecutive identical readings)"""
        anomalies = []
        consecutive_count = 1
        
        for i in range(1, len(values)):
            if abs(values[i] - values[i-1]) <= self.stuck_threshold:
                consecutive_count += 1
            else:
                if consecutive_count >= 5:  # 5 or more consecutive identical readings
                    # Mark all stuck readings as anomalies
                    for j in range(i - consecutive_count, i):
                        anomalies.append({
                            'timestamp': timestamps[j],
                            sensor_type: values[j],
                            'anomaly_type': 'stuck_sensor',
                            'anomaly_score': consecutive_count / 10.0,
                            'method': 'pattern'
                        })
                consecutive_count = 1
        
        # Check final sequence
        if consecutive_count >= 5:
            for j in range(len(values) - consecutive_count, len(values)):
                anomalies.append({
                    'timestamp': timestamps[j],
                    sensor_type: values[j],
                    'anomaly_type': 'stuck_sensor',
                    'anomaly_score': consecutive_count / 10.0,
                    'method': 'pattern'
                })
        
        return anomalies
    
    def _detect_spikes_and_dips(self, timestamps, values, sensor_type):
        """Detect sudden spikes and dips in sensor readings"""
        anomalies = []
        
        if len(values) < 3:
            return anomalies
        
        # Calculate rolling statistics
        window_size = min(10, len(values) // 3)
        rolling_mean = pd.Series(values).rolling(window=window_size, center=True).mean()
        rolling_std = pd.Series(values).rolling(window=window_size, center=True).std()
        
        for i in range(len(values)):
            if pd.notna(rolling_mean.iloc[i]) and pd.notna(rolling_std.iloc[i]):
                deviation = abs(values[i] - rolling_mean.iloc[i])
                threshold = self.spike_threshold * rolling_std.iloc[i]
                
                if deviation > threshold and rolling_std.iloc[i] > 0:
                    anomaly_type = 'spike' if values[i] > rolling_mean.iloc[i] else 'dip'
                    anomalies.append({
                        'timestamp': timestamps[i],
                        sensor_type: values[i],
                        'anomaly_type': anomaly_type,
                        'anomaly_score': deviation / rolling_std.iloc[i],
                        'method': 'pattern'
                    })
        
        return anomalies
    
    def _detect_trend_deviations(self, timestamps, values, sensor_type):
        """Detect deviations from expected trend"""
        anomalies = []
        
        if len(values) < 10:
            return anomalies
        
        # Calculate expected trend using linear regression
        x = np.arange(len(values))
        slope, intercept = np.polyfit(x, values, 1)
        expected_values = slope * x + intercept
        
        # Calculate residuals
        residuals = values - expected_values
        residual_std = np.std(residuals)
        
        # Find points that deviate significantly from trend
        for i, residual in enumerate(residuals):
            if abs(residual) > 3 * residual_std and residual_std > 0:
                anomalies.append({
                    'timestamp': timestamps[i],
                    sensor_type: values[i],
                    'anomaly_type': 'trend_deviation',
                    'anomaly_score': abs(residual) / residual_std,
                    'method': 'pattern'
                })
        
        return anomalies
    
    def _create_features(self, data_df, sensor_type):
        """Create features for ML-based anomaly detection"""
        values = data_df[sensor_type].values
        features = []
        
        for i in range(len(values)):
            feature_vector = [values[i]]
            
            # Add lag features
            for lag in [1, 2, 3]:
                if i >= lag:
                    feature_vector.append(values[i - lag])
                else:
                    feature_vector.append(values[i])
            
            # Add rolling statistics
            window_start = max(0, i - 4)
            window_values = values[window_start:i+1]
            
            feature_vector.extend([
                np.mean(window_values),
                np.std(window_values) if len(window_values) > 1 else 0,
                np.min(window_values),
                np.max(window_values)
            ])
            
            # Add rate of change
            if i > 0:
                feature_vector.append(values[i] - values[i-1])
            else:
                feature_vector.append(0)
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def predict_anomaly(self, new_data, sensor_type):
        """Predict if new data point is anomalous using fitted model"""
        if sensor_type not in self.fitted_models:
            return False, 0.0
        
        try:
            model_info = self.fitted_models[sensor_type]
            isolation_forest = model_info['isolation_forest']
            scaler = model_info['scaler']
            
            # Create features for new data point
            features = np.array([[new_data[sensor_type]]])  # Simplified for real-time
            scaled_features = scaler.transform(features)
            
            prediction = isolation_forest.predict(scaled_features)[0]
            score = abs(isolation_forest.decision_function(scaled_features)[0])
            
            return prediction == -1, score
        
        except Exception as e:
            print(f"Error in anomaly prediction: {e}")
            return False, 0.0
    
    def update_thresholds(self, z_score_threshold=None, iqr_multiplier=None, 
                         stuck_threshold=None, spike_threshold=None):
        """Update anomaly detection thresholds"""
        if z_score_threshold is not None:
            self.z_score_threshold = z_score_threshold
        if iqr_multiplier is not None:
            self.iqr_multiplier = iqr_multiplier
        if stuck_threshold is not None:
            self.stuck_threshold = stuck_threshold
        if spike_threshold is not None:
            self.spike_threshold = spike_threshold
    
    def get_anomaly_summary(self, anomalies_df):
        """Get summary statistics of detected anomalies"""
        if anomalies_df.empty:
            return {}
        
        summary = {
            'total_anomalies': len(anomalies_df),
            'by_type': anomalies_df['anomaly_type'].value_counts().to_dict(),
            'by_method': anomalies_df['method'].value_counts().to_dict(),
            'average_score': anomalies_df['anomaly_score'].mean(),
            'max_score': anomalies_df['anomaly_score'].max(),
            'time_range': {
                'start': anomalies_df['timestamp'].min(),
                'end': anomalies_df['timestamp'].max()
            }
        }
        
        return summary
