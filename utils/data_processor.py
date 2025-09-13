import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import signal, stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """Utility class for processing and analyzing sensor data"""
    
    def __init__(self):
        self.scalers = {}
        self.processing_history = []
    
    def clean_sensor_data(self, data_df: pd.DataFrame, 
                         remove_outliers: bool = True,
                         fill_missing: bool = True,
                         smooth_data: bool = False) -> pd.DataFrame:
        """Clean sensor data by removing outliers, filling missing values, and smoothing"""
        cleaned_df = data_df.copy()
        
        # Ensure timestamp is datetime
        if 'timestamp' in cleaned_df.columns:
            cleaned_df['timestamp'] = pd.to_datetime(cleaned_df['timestamp'])
            cleaned_df = cleaned_df.sort_values('timestamp')
        
        sensor_columns = ['temperature', 'pressure', 'vibration', 'humidity']
        
        for sensor in sensor_columns:
            if sensor in cleaned_df.columns:
                # Remove outliers using IQR method
                if remove_outliers:
                    cleaned_df[sensor] = self._remove_outliers_iqr(cleaned_df[sensor])
                
                # Fill missing values
                if fill_missing:
                    cleaned_df[sensor] = self._fill_missing_values(cleaned_df[sensor])
                
                # Smooth data
                if smooth_data:
                    cleaned_df[sensor] = self._smooth_series(cleaned_df[sensor])
        
        return cleaned_df
    
    def _remove_outliers_iqr(self, series: pd.Series, multiplier: float = 1.5) -> pd.Series:
        """Remove outliers using Interquartile Range method"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        # Replace outliers with NaN
        cleaned_series = series.copy()
        cleaned_series[(series < lower_bound) | (series > upper_bound)] = np.nan
        
        return cleaned_series
    
    def _fill_missing_values(self, series: pd.Series, method: str = 'interpolate') -> pd.Series:
        """Fill missing values using specified method"""
        if method == 'interpolate':
            return series.interpolate(method='linear', limit_direction='both')
        elif method == 'forward_fill':
            return series.fillna(method='ffill')
        elif method == 'backward_fill':
            return series.fillna(method='bfill')
        elif method == 'mean':
            return series.fillna(series.mean())
        else:
            return series
    
    def _smooth_series(self, series: pd.Series, window: int = 5) -> pd.Series:
        """Smooth time series using rolling average"""
        return series.rolling(window=window, center=True).mean().fillna(series)
    
    def calculate_statistics(self, data_df: pd.DataFrame, 
                           sensor_type: Optional[str] = None) -> Dict[str, Any]:
        """Calculate comprehensive statistics for sensor data"""
        if data_df.empty:
            return {}
        
        stats_dict = {}
        
        sensor_columns = [sensor_type] if sensor_type else ['temperature', 'pressure', 'vibration', 'humidity']
        
        for sensor in sensor_columns:
            if sensor in data_df.columns:
                values = data_df[sensor].dropna()
                
                if len(values) > 0:
                    stats_dict[sensor] = {
                        'count': len(values),
                        'mean': float(values.mean()),
                        'median': float(values.median()),
                        'std': float(values.std()),
                        'variance': float(values.var()),
                        'min': float(values.min()),
                        'max': float(values.max()),
                        'range': float(values.max() - values.min()),
                        'skewness': float(values.skew()),
                        'kurtosis': float(values.kurtosis()),
                        'q25': float(values.quantile(0.25)),
                        'q75': float(values.quantile(0.75)),
                        'iqr': float(values.quantile(0.75) - values.quantile(0.25)),
                        'coefficient_of_variation': float(values.std() / values.mean() * 100) if values.mean() != 0 else 0
                    }
        
        # Add temporal statistics if timestamp is available
        if 'timestamp' in data_df.columns:
            timestamps = pd.to_datetime(data_df['timestamp'])
            stats_dict['temporal'] = {
                'start_time': timestamps.min().isoformat(),
                'end_time': timestamps.max().isoformat(),
                'duration': str(timestamps.max() - timestamps.min()),
                'data_points': len(timestamps),
                'avg_interval': str(timestamps.diff().median()) if len(timestamps) > 1 else 'N/A'
            }
        
        return stats_dict
    
    def detect_changepoints(self, data_df: pd.DataFrame, sensor_type: str, 
                          min_size: int = 10) -> List[Dict[str, Any]]:
        """Detect change points in sensor data"""
        if sensor_type not in data_df.columns or len(data_df) < min_size * 2:
            return []
        
        values = data_df[sensor_type].dropna().values
        timestamps = pd.to_datetime(data_df['timestamp']).iloc[:len(values)]
        
        changepoints = []
        
        # Simple change point detection using sliding window variance
        window_size = max(min_size, len(values) // 10)
        
        for i in range(window_size, len(values) - window_size):
            before_window = values[i-window_size:i]
            after_window = values[i:i+window_size]
            
            # Statistical test for mean difference
            t_stat, p_value = stats.ttest_ind(before_window, after_window)
            
            if p_value < 0.05:  # Significant change detected
                mean_before = np.mean(before_window)
                mean_after = np.mean(after_window)
                
                changepoints.append({
                    'timestamp': timestamps.iloc[i],
                    'index': i,
                    'mean_before': mean_before,
                    'mean_after': mean_after,
                    'change_magnitude': abs(mean_after - mean_before),
                    'p_value': p_value,
                    't_statistic': abs(t_stat),
                    'direction': 'increase' if mean_after > mean_before else 'decrease'
                })
        
        return changepoints
    
    def calculate_correlation_matrix(self, data_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation matrix between sensors"""
        sensor_columns = ['temperature', 'pressure', 'vibration', 'humidity']
        available_sensors = [col for col in sensor_columns if col in data_df.columns]
        
        if len(available_sensors) < 2:
            return pd.DataFrame()
        
        correlation_data = data_df[available_sensors].dropna()
        
        if correlation_data.empty:
            return pd.DataFrame()
        
        return correlation_data.corr()
    
    def perform_frequency_analysis(self, data_df: pd.DataFrame, sensor_type: str) -> Dict[str, Any]:
        """Perform frequency domain analysis on sensor data"""
        if sensor_type not in data_df.columns or len(data_df) < 10:
            return {}
        
        values = data_df[sensor_type].dropna().values
        
        if len(values) < 10:
            return {}
        
        # Calculate sampling frequency
        timestamps = pd.to_datetime(data_df['timestamp']).iloc[:len(values)]
        if len(timestamps) > 1:
            avg_interval = (timestamps.diff().median()).total_seconds()
            fs = 1.0 / avg_interval if avg_interval > 0 else 1.0
        else:
            fs = 1.0
        
        # Perform FFT
        fft_values = np.fft.fft(values)
        fft_freq = np.fft.fftfreq(len(values), 1/fs)
        
        # Get positive frequencies only
        positive_freq_idx = fft_freq > 0
        frequencies = fft_freq[positive_freq_idx]
        magnitudes = np.abs(fft_values[positive_freq_idx])
        
        # Find dominant frequencies
        peak_indices = signal.find_peaks(magnitudes, height=np.max(magnitudes) * 0.1)[0]
        
        dominant_frequencies = []
        for idx in peak_indices[:5]:  # Top 5 peaks
            dominant_frequencies.append({
                'frequency': float(frequencies[idx]),
                'magnitude': float(magnitudes[idx]),
                'period': float(1/frequencies[idx]) if frequencies[idx] != 0 else float('inf')
            })
        
        return {
            'sampling_frequency': fs,
            'nyquist_frequency': fs / 2,
            'dominant_frequencies': sorted(dominant_frequencies, 
                                         key=lambda x: x['magnitude'], reverse=True),
            'total_energy': float(np.sum(magnitudes**2)),
            'spectral_centroid': float(np.sum(frequencies * magnitudes) / np.sum(magnitudes)) if np.sum(magnitudes) > 0 else 0
        }
    
    def calculate_data_quality_score(self, data_df: pd.DataFrame, sensor_type: str) -> Dict[str, Any]:
        """Calculate comprehensive data quality score"""
        if sensor_type not in data_df.columns:
            return {'quality_score': 0, 'issues': ['Sensor data not available']}
        
        values = data_df[sensor_type]
        timestamps = pd.to_datetime(data_df['timestamp'])
        
        quality_metrics = {}
        issues = []
        penalties = 0
        
        # Completeness check
        missing_ratio = values.isna().sum() / len(values)
        quality_metrics['completeness'] = (1 - missing_ratio) * 100
        
        if missing_ratio > 0.1:
            issues.append(f"High missing data ratio: {missing_ratio:.2%}")
            penalties += missing_ratio * 30
        
        # Outlier check
        clean_values = values.dropna()
        if len(clean_values) > 0:
            z_scores = np.abs(stats.zscore(clean_values))
            outlier_ratio = np.sum(z_scores > 3) / len(clean_values)
            quality_metrics['outlier_free'] = (1 - outlier_ratio) * 100
            
            if outlier_ratio > 0.05:
                issues.append(f"High outlier ratio: {outlier_ratio:.2%}")
                penalties += outlier_ratio * 25
        
        # Consistency check (stuck values)
        if len(clean_values) > 1:
            stuck_ratio = 0
            consecutive_identical = 1
            max_consecutive = 1
            
            for i in range(1, len(clean_values)):
                if abs(clean_values.iloc[i] - clean_values.iloc[i-1]) < 0.001:
                    consecutive_identical += 1
                    max_consecutive = max(max_consecutive, consecutive_identical)
                else:
                    consecutive_identical = 1
            
            stuck_ratio = max_consecutive / len(clean_values)
            quality_metrics['consistency'] = (1 - stuck_ratio) * 100
            
            if stuck_ratio > 0.1:
                issues.append(f"Sensor appears stuck: {max_consecutive} consecutive identical values")
                penalties += stuck_ratio * 20
        
        # Temporal consistency check
        if len(timestamps) > 1:
            time_diffs = timestamps.diff().dropna()
            if len(time_diffs) > 0:
                median_interval = time_diffs.median()
                large_gaps = time_diffs > median_interval * 3
                gap_ratio = large_gaps.sum() / len(time_diffs)
                
                quality_metrics['temporal_consistency'] = (1 - gap_ratio) * 100
                
                if gap_ratio > 0.05:
                    issues.append(f"Irregular timing: {gap_ratio:.2%} large gaps")
                    penalties += gap_ratio * 15
        
        # Noise level check
        if len(clean_values) > 10:
            # Calculate signal-to-noise ratio
            rolling_mean = clean_values.rolling(window=5, center=True).mean()
            noise = clean_values - rolling_mean
            signal_power = np.var(rolling_mean.dropna())
            noise_power = np.var(noise.dropna())
            
            if noise_power > 0 and signal_power > 0:
                snr = 10 * np.log10(signal_power / noise_power)
                quality_metrics['signal_to_noise_ratio'] = snr
                
                if snr < 10:  # SNR less than 10 dB
                    issues.append(f"Low signal-to-noise ratio: {snr:.1f} dB")
                    penalties += 10
        
        # Calculate overall quality score
        base_score = 100
        final_score = max(0, base_score - penalties)
        
        quality_metrics['overall_score'] = final_score
        
        # Classify quality
        if final_score >= 90:
            quality_class = 'Excellent'
        elif final_score >= 80:
            quality_class = 'Good'
        elif final_score >= 70:
            quality_class = 'Fair'
        elif final_score >= 60:
            quality_class = 'Poor'
        else:
            quality_class = 'Very Poor'
        
        return {
            'quality_score': final_score,
            'quality_class': quality_class,
            'metrics': quality_metrics,
            'issues': issues,
            'recommendations': self._generate_quality_recommendations(issues, quality_metrics)
        }
    
    def _generate_quality_recommendations(self, issues: List[str], metrics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on quality issues"""
        recommendations = []
        
        if any('missing' in issue.lower() for issue in issues):
            recommendations.append("Implement data interpolation or sensor redundancy")
        
        if any('outlier' in issue.lower() for issue in issues):
            recommendations.append("Apply outlier detection and filtering algorithms")
        
        if any('stuck' in issue.lower() for issue in issues):
            recommendations.append("Check sensor hardware and recalibrate if necessary")
        
        if any('timing' in issue.lower() or 'gap' in issue.lower() for issue in issues):
            recommendations.append("Verify data acquisition system timing and connectivity")
        
        if any('noise' in issue.lower() or 'signal' in issue.lower() for issue in issues):
            recommendations.append("Apply noise filtering and improve sensor shielding")
        
        if not recommendations:
            recommendations.append("Data quality is acceptable - continue monitoring")
        
        return recommendations
    
    def resample_data(self, data_df: pd.DataFrame, frequency: str = '1min', 
                     aggregation: str = 'mean') -> pd.DataFrame:
        """Resample sensor data to specified frequency"""
        if 'timestamp' not in data_df.columns:
            return data_df
        
        resampled_df = data_df.copy()
        resampled_df['timestamp'] = pd.to_datetime(resampled_df['timestamp'])
        resampled_df = resampled_df.set_index('timestamp')
        
        sensor_columns = ['temperature', 'pressure', 'vibration', 'humidity']
        available_sensors = [col for col in sensor_columns if col in resampled_df.columns]
        
        if aggregation == 'mean':
            resampled = resampled_df[available_sensors].resample(frequency).mean()
        elif aggregation == 'median':
            resampled = resampled_df[available_sensors].resample(frequency).median()
        elif aggregation == 'min':
            resampled = resampled_df[available_sensors].resample(frequency).min()
        elif aggregation == 'max':
            resampled = resampled_df[available_sensors].resample(frequency).max()
        else:
            resampled = resampled_df[available_sensors].resample(frequency).mean()
        
        resampled = resampled.reset_index()
        return resampled
    
    def calculate_calibration_stats(self, data_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate calibration accuracy statistics"""
        stats = {}
        
        sensor_columns = ['temperature', 'pressure', 'vibration', 'humidity']
        
        for sensor in sensor_columns:
            if sensor in data_df.columns:
                values = data_df[sensor].dropna()
                
                if len(values) > 10:
                    # Simulate reference values for calibration accuracy calculation
                    # In a real system, these would come from calibrated reference sensors
                    reference_values = values + np.random.normal(0, 0.1, len(values))
                    
                    # Calculate accuracy metrics
                    mae = np.mean(np.abs(values - reference_values))
                    rmse = np.sqrt(np.mean((values - reference_values)**2))
                    mape = np.mean(np.abs((values - reference_values) / reference_values)) * 100
                    
                    # Correlation with reference
                    correlation = np.corrcoef(values, reference_values)[0, 1]
                    
                    # Accuracy percentage (inverse of error)
                    mean_value = np.mean(np.abs(reference_values))
                    if mean_value > 0:
                        accuracy = max(0, 100 - (mae / mean_value * 100))
                    else:
                        accuracy = 100
                    
                    stats[sensor] = {
                        'accuracy': accuracy,
                        'mae': mae,
                        'rmse': rmse,
                        'mape': mape,
                        'correlation': correlation,
                        'bias': np.mean(values - reference_values),
                        'precision': np.std(values - reference_values)
                    }
        
        return stats
    
    def normalize_data(self, data_df: pd.DataFrame, method: str = 'minmax', 
                      sensor_type: Optional[str] = None) -> pd.DataFrame:
        """Normalize sensor data using specified method"""
        normalized_df = data_df.copy()
        
        sensor_columns = [sensor_type] if sensor_type else ['temperature', 'pressure', 'vibration', 'humidity']
        available_sensors = [col for col in sensor_columns if col in normalized_df.columns]
        
        for sensor in available_sensors:
            values = normalized_df[sensor].dropna()
            
            if len(values) > 0:
                if method == 'minmax':
                    if sensor not in self.scalers:
                        self.scalers[sensor] = MinMaxScaler()
                    
                    # Fit and transform
                    scaled_values = self.scalers[sensor].fit_transform(values.values.reshape(-1, 1)).flatten()
                    
                elif method == 'zscore':
                    if sensor not in self.scalers:
                        self.scalers[sensor] = StandardScaler()
                    
                    scaled_values = self.scalers[sensor].fit_transform(values.values.reshape(-1, 1)).flatten()
                
                elif method == 'robust':
                    # Robust scaling using median and IQR
                    median = values.median()
                    q75 = values.quantile(0.75)
                    q25 = values.quantile(0.25)
                    iqr = q75 - q25
                    
                    if iqr > 0:
                        scaled_values = (values - median) / iqr
                    else:
                        scaled_values = values - median
                
                else:
                    scaled_values = values  # No scaling
                
                # Update the dataframe
                mask = normalized_df[sensor].notna()
                normalized_df.loc[mask, sensor] = scaled_values
        
        return normalized_df
    
    def calculate_sensor_health_score(self, data_df: pd.DataFrame, sensor_type: str) -> Dict[str, Any]:
        """Calculate overall health score for a sensor"""
        if sensor_type not in data_df.columns:
            return {'health_score': 0, 'status': 'No Data'}
        
        # Get quality metrics
        quality_result = self.calculate_data_quality_score(data_df, sensor_type)
        quality_score = quality_result['quality_score']
        
        # Calculate additional health metrics
        values = data_df[sensor_type].dropna()
        health_factors = []
        
        # Data availability
        availability = len(values) / len(data_df) * 100
        health_factors.append(availability * 0.2)  # 20% weight
        
        # Quality score
        health_factors.append(quality_score * 0.4)  # 40% weight
        
        # Stability (low variance relative to mean)
        if len(values) > 1:
            cv = values.std() / values.mean() * 100 if values.mean() != 0 else 100
            stability_score = max(0, 100 - cv)
            health_factors.append(stability_score * 0.2)  # 20% weight
        
        # Responsiveness (rate of change is reasonable)
        if len(values) > 5:
            changes = np.abs(np.diff(values))
            avg_change = np.mean(changes)
            max_change = np.max(changes)
            
            # Responsive but not erratic
            if max_change > 0:
                responsiveness = min(100, (avg_change / max_change) * 100)
            else:
                responsiveness = 100
            
            health_factors.append(responsiveness * 0.2)  # 20% weight
        
        # Calculate overall health score
        overall_health = sum(health_factors)
        
        # Determine status
        if overall_health >= 90:
            status = 'Excellent'
        elif overall_health >= 80:
            status = 'Good'
        elif overall_health >= 70:
            status = 'Fair'
        elif overall_health >= 60:
            status = 'Poor'
        else:
            status = 'Critical'
        
        return {
            'health_score': overall_health,
            'status': status,
            'availability': availability,
            'quality_score': quality_score,
            'factors': {
                'availability': availability,
                'quality': quality_score,
                'stability': health_factors[2] / 0.2 if len(health_factors) > 2 else 0,
                'responsiveness': health_factors[3] / 0.2 if len(health_factors) > 3 else 0
            }
        }
