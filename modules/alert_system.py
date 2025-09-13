import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json

class AlertSystem:
    """Manages threshold-based alerts and notifications for sensor monitoring"""
    
    def __init__(self):
        self.active_alerts = []
        self.alert_history = []
        self.alert_rules = {}
        self.notification_settings = {
            'enable_email': False,
            'enable_dashboard': True,
            'enable_logging': True,
            'severity_levels': ['info', 'warning', 'critical']
        }
        
        # Default alert rules
        self.default_thresholds = {
            'temperature': {
                'min': 18.0, 'max': 35.0,
                'warning_buffer': 2.0,  # Warning zone before critical
                'rate_of_change_limit': 5.0  # Max change per minute
            },
            'pressure': {
                'min': 980.0, 'max': 1040.0,
                'warning_buffer': 10.0,
                'rate_of_change_limit': 20.0
            },
            'vibration': {
                'min': 0.0, 'max': 10.0,
                'warning_buffer': 1.0,
                'rate_of_change_limit': 3.0
            },
            'humidity': {
                'min': 30.0, 'max': 80.0,
                'warning_buffer': 5.0,
                'rate_of_change_limit': 10.0
            }
        }
        
        # Initialize alert rules with defaults
        self.alert_rules = self.default_thresholds.copy()
    
    def check_thresholds(self, sensor_data: Dict[str, Any], thresholds: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        """Check sensor data against thresholds and generate alerts"""
        alerts = []
        timestamp = sensor_data.get('timestamp', datetime.now())
        
        for sensor_type in ['temperature', 'pressure', 'vibration', 'humidity']:
            if sensor_type in sensor_data and sensor_type in thresholds:
                value = sensor_data[sensor_type]
                sensor_thresholds = thresholds[sensor_type]
                
                # Check for threshold violations
                alert = self._check_sensor_threshold(
                    sensor_type, value, sensor_thresholds, timestamp
                )
                
                if alert:
                    alerts.append(alert)
                    self._add_alert_to_history(alert)
        
        return alerts
    
    def _check_sensor_threshold(self, sensor_type: str, value: float, 
                               thresholds: Dict[str, float], timestamp: datetime) -> Optional[Dict[str, Any]]:
        """Check individual sensor against thresholds"""
        min_val = thresholds['min']
        max_val = thresholds['max']
        warning_buffer = thresholds.get('warning_buffer', 0.0)
        
        # Critical alerts (outside limits)
        if value < min_val:
            return {
                'timestamp': timestamp,
                'sensor_type': sensor_type,
                'level': 'critical',
                'type': 'threshold_violation',
                'message': f"{sensor_type.title()} critically low: {value:.2f} (min: {min_val})",
                'value': value,
                'threshold': min_val,
                'severity_score': abs(value - min_val) / abs(min_val) * 100
            }
        
        elif value > max_val:
            return {
                'timestamp': timestamp,
                'sensor_type': sensor_type,
                'level': 'critical',
                'type': 'threshold_violation',
                'message': f"{sensor_type.title()} critically high: {value:.2f} (max: {max_val})",
                'value': value,
                'threshold': max_val,
                'severity_score': abs(value - max_val) / abs(max_val) * 100
            }
        
        # Warning alerts (approaching limits)
        elif value < min_val + warning_buffer:
            return {
                'timestamp': timestamp,
                'sensor_type': sensor_type,
                'level': 'warning',
                'type': 'approaching_threshold',
                'message': f"{sensor_type.title()} approaching low limit: {value:.2f} (warning at: {min_val + warning_buffer})",
                'value': value,
                'threshold': min_val + warning_buffer,
                'severity_score': abs(value - (min_val + warning_buffer)) / warning_buffer * 50
            }
        
        elif value > max_val - warning_buffer:
            return {
                'timestamp': timestamp,
                'sensor_type': sensor_type,
                'level': 'warning',
                'type': 'approaching_threshold',
                'message': f"{sensor_type.title()} approaching high limit: {value:.2f} (warning at: {max_val - warning_buffer})",
                'value': value,
                'threshold': max_val - warning_buffer,
                'severity_score': abs(value - (max_val - warning_buffer)) / warning_buffer * 50
            }
        
        return None
    
    def check_rate_of_change(self, sensor_data_history: pd.DataFrame, sensor_type: str) -> List[Dict[str, Any]]:
        """Check for rapid changes in sensor values"""
        alerts = []
        
        if sensor_type not in sensor_data_history.columns or len(sensor_data_history) < 2:
            return alerts
        
        if sensor_type not in self.alert_rules:
            return alerts
        
        recent_data = sensor_data_history.tail(10)  # Check last 10 readings
        values = recent_data[sensor_type].values
        timestamps = pd.to_datetime(recent_data['timestamp']).values
        
        rate_limit = self.alert_rules[sensor_type].get('rate_of_change_limit', float('inf'))
        
        for i in range(1, len(values)):
            time_diff = (timestamps[i] - timestamps[i-1]).astype('timedelta64[s]').astype(float) / 60  # minutes
            
            if time_diff > 0:
                rate_of_change = abs(values[i] - values[i-1]) / time_diff
                
                if rate_of_change > rate_limit:
                    alert = {
                        'timestamp': timestamps[i],
                        'sensor_type': sensor_type,
                        'level': 'warning',
                        'type': 'rapid_change',
                        'message': f"{sensor_type.title()} changing rapidly: {rate_of_change:.2f} units/min (limit: {rate_limit})",
                        'value': values[i],
                        'rate_of_change': rate_of_change,
                        'severity_score': (rate_of_change / rate_limit) * 75
                    }
                    
                    alerts.append(alert)
                    self._add_alert_to_history(alert)
        
        return alerts
    
    def check_sensor_health(self, sensor_data_history: pd.DataFrame) -> List[Dict[str, Any]]:
        """Check overall sensor health and detect issues"""
        alerts = []
        
        for sensor_type in ['temperature', 'pressure', 'vibration', 'humidity']:
            if sensor_type in sensor_data_history.columns:
                health_alerts = self._check_individual_sensor_health(
                    sensor_data_history, sensor_type
                )
                alerts.extend(health_alerts)
        
        return alerts
    
    def _check_individual_sensor_health(self, data: pd.DataFrame, sensor_type: str) -> List[Dict[str, Any]]:
        """Check health of individual sensor"""
        alerts = []
        
        if len(data) < 10:
            return alerts
        
        values = data[sensor_type].dropna()
        timestamps = pd.to_datetime(data['timestamp'])
        
        # Check for stuck sensor (consecutive identical values)
        stuck_threshold = 5  # Number of consecutive identical readings
        consecutive_count = 1
        
        for i in range(1, len(values)):
            if abs(values.iloc[i] - values.iloc[i-1]) < 0.001:
                consecutive_count += 1
            else:
                if consecutive_count >= stuck_threshold:
                    alert = {
                        'timestamp': timestamps.iloc[i-1],
                        'sensor_type': sensor_type,
                        'level': 'critical',
                        'type': 'sensor_stuck',
                        'message': f"{sensor_type.title()} sensor appears stuck ({consecutive_count} identical readings)",
                        'value': values.iloc[i-1],
                        'consecutive_count': consecutive_count,
                        'severity_score': min(100, consecutive_count * 10)
                    }
                    alerts.append(alert)
                consecutive_count = 1
        
        # Check for missing data (time gaps)
        if len(timestamps) > 1:
            time_diffs = timestamps.diff().dropna()
            median_interval = time_diffs.median()
            
            for i, diff in enumerate(time_diffs):
                if diff > median_interval * 3:  # Gap more than 3x normal interval
                    alert = {
                        'timestamp': timestamps.iloc[i+1],
                        'sensor_type': sensor_type,
                        'level': 'warning',
                        'type': 'data_gap',
                        'message': f"{sensor_type.title()} sensor data gap detected: {diff}",
                        'value': None,
                        'gap_duration': str(diff),
                        'severity_score': min(100, (diff / median_interval) * 20)
                    }
                    alerts.append(alert)
        
        # Check for excessive noise
        if len(values) >= 20:
            recent_values = values.tail(20)
            noise_level = recent_values.std()
            
            # Define noise thresholds based on sensor type
            noise_thresholds = {
                'temperature': 2.0,
                'pressure': 5.0,
                'vibration': 1.0,
                'humidity': 3.0
            }
            
            threshold = noise_thresholds.get(sensor_type, 1.0)
            
            if noise_level > threshold:
                alert = {
                    'timestamp': timestamps.iloc[-1],
                    'sensor_type': sensor_type,
                    'level': 'warning',
                    'type': 'excessive_noise',
                    'message': f"{sensor_type.title()} sensor showing excessive noise: σ = {noise_level:.3f}",
                    'value': values.iloc[-1],
                    'noise_level': noise_level,
                    'severity_score': min(100, (noise_level / threshold) * 60)
                }
                alerts.append(alert)
        
        return alerts
    
    def _add_alert_to_history(self, alert: Dict[str, Any]):
        """Add alert to history and manage active alerts"""
        # Add to history
        self.alert_history.append(alert)
        
        # Manage active alerts
        alert_key = f"{alert['sensor_type']}_{alert['type']}"
        
        # Remove old alerts of same type for same sensor
        self.active_alerts = [
            a for a in self.active_alerts 
            if not (a['sensor_type'] == alert['sensor_type'] and a['type'] == alert['type'])
        ]
        
        # Add new alert to active list
        self.active_alerts.append(alert)
        
        # Keep only recent alerts in history (last 1000)
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]
    
    def get_active_alerts(self, severity_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get currently active alerts"""
        if severity_filter:
            return [alert for alert in self.active_alerts if alert['level'] == severity_filter]
        return self.active_alerts.copy()
    
    def get_alert_history(self, hours_back: int = 24, sensor_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get alert history for specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        filtered_alerts = [
            alert for alert in self.alert_history
            if alert['timestamp'] >= cutoff_time
        ]
        
        if sensor_type:
            filtered_alerts = [
                alert for alert in filtered_alerts
                if alert['sensor_type'] == sensor_type
            ]
        
        return filtered_alerts
    
    def update_alert_rules(self, new_rules: Dict[str, Dict[str, float]]):
        """Update alert thresholds and rules"""
        for sensor_type, rules in new_rules.items():
            if sensor_type in self.alert_rules:
                self.alert_rules[sensor_type].update(rules)
            else:
                self.alert_rules[sensor_type] = rules.copy()
    
    def clear_alerts(self, sensor_type: Optional[str] = None, alert_type: Optional[str] = None):
        """Clear active alerts based on filters"""
        if sensor_type and alert_type:
            self.active_alerts = [
                alert for alert in self.active_alerts
                if not (alert['sensor_type'] == sensor_type and alert['type'] == alert_type)
            ]
        elif sensor_type:
            self.active_alerts = [
                alert for alert in self.active_alerts
                if alert['sensor_type'] != sensor_type
            ]
        elif alert_type:
            self.active_alerts = [
                alert for alert in self.active_alerts
                if alert['type'] != alert_type
            ]
        else:
            self.active_alerts.clear()
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get statistics about alerts"""
        if not self.alert_history:
            return {
                'total_alerts': 0,
                'by_severity': {},
                'by_sensor': {},
                'by_type': {},
                'recent_trend': 'No data'
            }
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(self.alert_history)
        
        stats = {
            'total_alerts': len(df),
            'active_alerts': len(self.active_alerts),
            'by_severity': df['level'].value_counts().to_dict(),
            'by_sensor': df['sensor_type'].value_counts().to_dict(),
            'by_type': df['type'].value_counts().to_dict()
        }
        
        # Calculate recent trend (last 24 hours vs previous 24 hours)
        now = datetime.now()
        recent_24h = df[pd.to_datetime(df['timestamp']) >= now - timedelta(hours=24)]
        previous_24h = df[
            (pd.to_datetime(df['timestamp']) >= now - timedelta(hours=48)) &
            (pd.to_datetime(df['timestamp']) < now - timedelta(hours=24))
        ]
        
        recent_count = len(recent_24h)
        previous_count = len(previous_24h)
        
        if previous_count == 0:
            trend = 'New alerts' if recent_count > 0 else 'No alerts'
        else:
            change_pct = ((recent_count - previous_count) / previous_count) * 100
            if change_pct > 10:
                trend = f'Increasing (+{change_pct:.1f}%)'
            elif change_pct < -10:
                trend = f'Decreasing ({change_pct:.1f}%)'
            else:
                trend = 'Stable'
        
        stats['recent_trend'] = trend
        stats['recent_24h_count'] = recent_count
        stats['previous_24h_count'] = previous_count
        
        return stats
    
    def export_alerts(self, format_type: str = 'json', hours_back: int = 24) -> str:
        """Export alert history in specified format"""
        alerts = self.get_alert_history(hours_back)
        
        if format_type.lower() == 'json':
            # Convert datetime objects to strings for JSON serialization
            alerts_serializable = []
            for alert in alerts:
                alert_copy = alert.copy()
                if isinstance(alert_copy['timestamp'], datetime):
                    alert_copy['timestamp'] = alert_copy['timestamp'].isoformat()
                alerts_serializable.append(alert_copy)
            
            return json.dumps(alerts_serializable, indent=2)
        
        elif format_type.lower() == 'csv':
            if alerts:
                df = pd.DataFrame(alerts)
                return df.to_csv(index=False)
            else:
                return "timestamp,sensor_type,level,type,message,value\n"
        
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def create_custom_alert_rule(self, name: str, sensor_type: str, condition: str, 
                                threshold: float, message_template: str):
        """Create a custom alert rule"""
        custom_rule = {
            'name': name,
            'sensor_type': sensor_type,
            'condition': condition,  # 'greater_than', 'less_than', 'equals', 'not_equals'
            'threshold': threshold,
            'message_template': message_template,
            'created_at': datetime.now(),
            'enabled': True
        }
        
        if 'custom_rules' not in self.alert_rules:
            self.alert_rules['custom_rules'] = []
        
        self.alert_rules['custom_rules'].append(custom_rule)
        return custom_rule
    
    def check_custom_rules(self, sensor_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check sensor data against custom alert rules"""
        alerts = []
        
        if 'custom_rules' not in self.alert_rules:
            return alerts
        
        timestamp = sensor_data.get('timestamp', datetime.now())
        
        for rule in self.alert_rules['custom_rules']:
            if not rule.get('enabled', True):
                continue
            
            sensor_type = rule['sensor_type']
            if sensor_type not in sensor_data:
                continue
            
            value = sensor_data[sensor_type]
            condition = rule['condition']
            threshold = rule['threshold']
            
            triggered = False
            
            if condition == 'greater_than' and value > threshold:
                triggered = True
            elif condition == 'less_than' and value < threshold:
                triggered = True
            elif condition == 'equals' and abs(value - threshold) < 0.001:
                triggered = True
            elif condition == 'not_equals' and abs(value - threshold) >= 0.001:
                triggered = True
            
            if triggered:
                message = rule['message_template'].format(
                    sensor=sensor_type,
                    value=value,
                    threshold=threshold
                )
                
                alert = {
                    'timestamp': timestamp,
                    'sensor_type': sensor_type,
                    'level': 'warning',  # Default level for custom rules
                    'type': 'custom_rule',
                    'message': message,
                    'value': value,
                    'rule_name': rule['name'],
                    'severity_score': 50  # Default severity for custom rules
                }
                
                alerts.append(alert)
                self._add_alert_to_history(alert)
        
        return alerts
