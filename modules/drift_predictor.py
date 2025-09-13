import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Try to import Prophet, fall back to simple linear prediction if not available
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

class DriftPredictor:
    """Predicts sensor drift using various time series forecasting methods"""
    
    def __init__(self):
        self.models = {}
        self.prediction_cache = {}
        self.min_data_points = 20
    
    def predict_drift(self, data_df, sensor_type, days=7, method='prophet'):
        """Predict sensor drift for specified number of days"""
        if sensor_type not in data_df.columns:
            return None
        
        if len(data_df) < self.min_data_points:
            return None
        
        # Prepare data
        ts_data = self._prepare_time_series_data(data_df, sensor_type)
        
        if method == 'prophet' and PROPHET_AVAILABLE:
            return self._predict_with_prophet(ts_data, sensor_type, days)
        elif method == 'arima':
            return self._predict_with_arima(ts_data, sensor_type, days)
        elif method == 'linear_trend':
            return self._predict_with_linear_trend(ts_data, sensor_type, days)
        else:
            # Fallback to linear trend if Prophet not available
            return self._predict_with_linear_trend(ts_data, sensor_type, days)
    
    def _prepare_time_series_data(self, data_df, sensor_type):
        """Prepare time series data for forecasting"""
        # Ensure data is sorted by timestamp
        data_sorted = data_df.sort_values('timestamp').copy()
        
        # Remove any duplicates
        data_sorted = data_sorted.drop_duplicates(subset=['timestamp'])
        
        # Create time series
        ts_data = pd.DataFrame({
            'ds': pd.to_datetime(data_sorted['timestamp']),
            'y': data_sorted[sensor_type]
        })
        
        # Remove outliers (beyond 3 standard deviations)
        mean_val = ts_data['y'].mean()
        std_val = ts_data['y'].std()
        ts_data = ts_data[np.abs(ts_data['y'] - mean_val) <= 3 * std_val]
        
        return ts_data
    
    def _predict_with_prophet(self, ts_data, sensor_type, days):
        """Predict using Facebook Prophet"""
        try:
            # Initialize and fit Prophet model
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=False,
                interval_width=0.95,
                changepoint_prior_scale=0.05
            )
            
            model.fit(ts_data)
            
            # Create future dataframe
            future = model.make_future_dataframe(periods=days, freq='H')
            forecast = model.predict(future)
            
            # Extract predictions for the forecast period
            forecast_start_idx = len(ts_data)
            future_forecast = forecast.iloc[forecast_start_idx:]
            
            # Calculate trend and confidence
            trend_slope = self._calculate_trend_slope(forecast['trend'].values)
            confidence = self._calculate_prediction_confidence(
                ts_data['y'].values, forecast['yhat'].iloc[:len(ts_data)].values
            )
            
            # Store model for future use
            self.models[sensor_type] = model
            
            return {
                'method': 'prophet',
                'forecast': future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict('records'),
                'trend_slope': trend_slope,
                'confidence': confidence,
                'forecast_days': days,
                'model_components': {
                    'trend': forecast['trend'].iloc[-1] - forecast['trend'].iloc[0],
                    'seasonal': forecast['weekly'].iloc[-days:].mean() if 'weekly' in forecast.columns else 0
                }
            }
        
        except Exception as e:
            print(f"Prophet prediction failed: {e}")
            return self._predict_with_linear_trend(ts_data, sensor_type, days)
    
    def _predict_with_arima(self, ts_data, sensor_type, days):
        """Predict using ARIMA model"""
        try:
            # Prepare data
            y = ts_data['y'].values
            
            # Automatically determine ARIMA parameters
            best_aic = float('inf')
            best_order = (1, 1, 1)
            
            # Grid search for best parameters (simplified)
            for p in range(0, 3):
                for d in range(0, 2):
                    for q in range(0, 3):
                        try:
                            model = ARIMA(y, order=(p, d, q))
                            fitted_model = model.fit()
                            if fitted_model.aic < best_aic:
                                best_aic = fitted_model.aic
                                best_order = (p, d, q)
                        except:
                            continue
            
            # Fit best model
            model = ARIMA(y, order=best_order)
            fitted_model = model.fit()
            
            # Generate forecast
            forecast_steps = days * 24  # Hourly predictions
            forecast = fitted_model.forecast(steps=forecast_steps)
            forecast_ci = fitted_model.get_forecast(steps=forecast_steps).conf_int()
            
            # Create forecast dataframe
            last_timestamp = ts_data['ds'].iloc[-1]
            future_dates = pd.date_range(
                start=last_timestamp + timedelta(hours=1),
                periods=forecast_steps,
                freq='H'
            )
            
            forecast_df = pd.DataFrame({
                'ds': future_dates,
                'yhat': forecast,
                'yhat_lower': forecast_ci.iloc[:, 0],
                'yhat_upper': forecast_ci.iloc[:, 1]
            })
            
            # Calculate metrics
            trend_slope = self._calculate_trend_slope(forecast)
            confidence = self._calculate_prediction_confidence(
                y[-len(forecast):] if len(y) >= len(forecast) else y,
                fitted_model.fittedvalues[-len(y):]
            )
            
            return {
                'method': 'arima',
                'forecast': forecast_df.to_dict('records'),
                'trend_slope': trend_slope,
                'confidence': confidence,
                'forecast_days': days,
                'model_params': best_order,
                'aic': best_aic
            }
        
        except Exception as e:
            print(f"ARIMA prediction failed: {e}")
            return self._predict_with_linear_trend(ts_data, sensor_type, days)
    
    def _predict_with_linear_trend(self, ts_data, sensor_type, days):
        """Predict using simple linear trend"""
        try:
            # Convert timestamps to numeric for regression
            ts_data = ts_data.copy()
            ts_data['ds_numeric'] = pd.to_numeric(ts_data['ds'])
            
            # Fit linear regression
            X = ts_data['ds_numeric'].values.reshape(-1, 1)
            y = ts_data['y'].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            # Generate future timestamps
            last_timestamp = ts_data['ds'].iloc[-1]
            future_dates = pd.date_range(
                start=last_timestamp + timedelta(hours=1),
                periods=days * 24,
                freq='H'
            )
            
            future_numeric = pd.to_numeric(future_dates).values.reshape(-1, 1)
            future_predictions = model.predict(future_numeric)
            
            # Estimate confidence intervals (simple approach)
            residuals = y - model.predict(X)
            rmse = np.sqrt(mean_squared_error(y, model.predict(X)))
            
            forecast_df = pd.DataFrame({
                'ds': future_dates,
                'yhat': future_predictions,
                'yhat_lower': future_predictions - 1.96 * rmse,
                'yhat_upper': future_predictions + 1.96 * rmse
            })
            
            # Calculate metrics
            trend_slope = model.coef_[0] * (24 * 3600 * 1000000000)  # Convert to per day
            confidence = self._calculate_prediction_confidence(y, model.predict(X))
            
            return {
                'method': 'linear_trend',
                'forecast': forecast_df.to_dict('records'),
                'trend_slope': trend_slope,
                'confidence': confidence,
                'forecast_days': days,
                'slope': model.coef_[0],
                'intercept': model.intercept_,
                'rmse': rmse
            }
        
        except Exception as e:
            print(f"Linear trend prediction failed: {e}")
            return None
    
    def _calculate_trend_slope(self, values):
        """Calculate trend slope from time series values"""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        return slope
    
    def _calculate_prediction_confidence(self, actual, predicted):
        """Calculate confidence score for predictions"""
        if len(actual) != len(predicted) or len(actual) == 0:
            return 0.0
        
        # Calculate R-squared
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        
        if ss_tot == 0:
            return 100.0 if ss_res == 0 else 0.0
        
        r_squared = 1 - (ss_res / ss_tot)
        confidence = max(0, min(100, r_squared * 100))
        
        return confidence
    
    def detect_drift_patterns(self, data_df, sensor_type, window_size=100):
        """Detect different types of drift patterns"""
        if sensor_type not in data_df.columns or len(data_df) < window_size:
            return {}
        
        values = data_df[sensor_type].values
        timestamps = data_df['timestamp'].values
        
        patterns = {}
        
        # Gradual drift detection
        gradual_drift = self._detect_gradual_drift(values, window_size)
        patterns['gradual_drift'] = gradual_drift
        
        # Sudden drift detection
        sudden_drift = self._detect_sudden_drift(values, window_size)
        patterns['sudden_drift'] = sudden_drift
        
        # Periodic drift detection
        periodic_drift = self._detect_periodic_drift(values)
        patterns['periodic_drift'] = periodic_drift
        
        # Overall trend analysis
        overall_trend = self._analyze_overall_trend(values, timestamps)
        patterns['overall_trend'] = overall_trend
        
        return patterns
    
    def _detect_gradual_drift(self, values, window_size):
        """Detect gradual drift in sensor readings"""
        if len(values) < 2 * window_size:
            return {'detected': False}
        
        # Compare slopes of different segments
        mid_point = len(values) // 2
        
        # First half slope
        x1 = np.arange(mid_point)
        slope1, _ = np.polyfit(x1, values[:mid_point], 1)
        
        # Second half slope
        x2 = np.arange(len(values) - mid_point)
        slope2, _ = np.polyfit(x2, values[mid_point:], 1)
        
        # Check if there's significant change in slope
        slope_change = abs(slope2 - slope1)
        threshold = np.std(values) / len(values) * 10  # Adaptive threshold
        
        return {
            'detected': slope_change > threshold,
            'slope_change': slope_change,
            'first_half_slope': slope1,
            'second_half_slope': slope2,
            'significance': slope_change / threshold if threshold > 0 else 0
        }
    
    def _detect_sudden_drift(self, values, window_size):
        """Detect sudden changes in sensor readings"""
        change_points = []
        
        for i in range(window_size, len(values) - window_size):
            before = values[i-window_size:i]
            after = values[i:i+window_size]
            
            mean_before = np.mean(before)
            mean_after = np.mean(after)
            
            # Statistical test for significant difference
            std_before = np.std(before)
            std_after = np.std(after)
            pooled_std = np.sqrt((std_before**2 + std_after**2) / 2)
            
            if pooled_std > 0:
                t_stat = abs(mean_after - mean_before) / (pooled_std * np.sqrt(2/window_size))
                
                # Simple threshold (could use t-distribution)
                if t_stat > 2.0:  # Roughly 95% confidence
                    change_points.append({
                        'index': i,
                        'magnitude': mean_after - mean_before,
                        't_statistic': t_stat
                    })
        
        return {
            'detected': len(change_points) > 0,
            'change_points': change_points,
            'num_changes': len(change_points)
        }
    
    def _detect_periodic_drift(self, values):
        """Detect periodic patterns in drift"""
        if len(values) < 50:
            return {'detected': False}
        
        try:
            # Simple autocorrelation approach
            autocorr = np.correlate(values, values, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            
            # Find peaks in autocorrelation (excluding lag 0)
            peaks = []
            for i in range(1, min(len(autocorr) // 4, 100)):
                if (autocorr[i] > autocorr[i-1] and 
                    autocorr[i] > autocorr[i+1] and
                    autocorr[i] > 0.3 * autocorr[0]):  # 30% of max correlation
                    peaks.append((i, autocorr[i]))
            
            return {
                'detected': len(peaks) > 0,
                'periods': [p[0] for p in peaks],
                'correlations': [p[1] for p in peaks],
                'strongest_period': peaks[0][0] if peaks else None
            }
        
        except Exception:
            return {'detected': False}
    
    def _analyze_overall_trend(self, values, timestamps):
        """Analyze overall trend in the data"""
        if len(values) < 3:
            return {}
        
        # Linear trend
        x = np.arange(len(values))
        slope, intercept = np.polyfit(x, values, 1)
        
        # Trend significance
        residuals = values - (slope * x + intercept)
        r_squared = 1 - (np.sum(residuals**2) / np.sum((values - np.mean(values))**2))
        
        # Classify trend
        trend_type = 'stable'
        if abs(slope) > np.std(values) / len(values):
            trend_type = 'increasing' if slope > 0 else 'decreasing'
        
        return {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared,
            'trend_type': trend_type,
            'trend_strength': abs(slope) * len(values) / np.std(values) if np.std(values) > 0 else 0
        }
    
    def get_prediction_summary(self, prediction_result):
        """Get summary of prediction results"""
        if not prediction_result:
            return {}
        
        forecast_data = prediction_result['forecast']
        if not forecast_data:
            return {}
        
        predicted_values = [point['yhat'] for point in forecast_data]
        
        return {
            'method': prediction_result['method'],
            'forecast_days': prediction_result['forecast_days'],
            'trend_slope': prediction_result['trend_slope'],
            'confidence': prediction_result['confidence'],
            'predicted_range': {
                'min': min(predicted_values),
                'max': max(predicted_values),
                'mean': np.mean(predicted_values)
            },
            'drift_magnitude': predicted_values[-1] - predicted_values[0] if len(predicted_values) > 1 else 0,
            'forecast_points': len(predicted_values)
        }
