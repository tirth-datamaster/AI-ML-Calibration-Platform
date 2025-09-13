import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import os
import json
import threading

class DatabaseManager:
    """Manages SQLite database operations for sensor data storage"""
    
    def __init__(self, db_path='sensor_data.db'):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Sensor data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sensor_readings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    temperature REAL,
                    pressure REAL,
                    vibration REAL,
                    humidity REAL,
                    is_calibrated BOOLEAN DEFAULT 1,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Anomalies table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS anomalies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    sensor_type TEXT NOT NULL,
                    sensor_value REAL NOT NULL,
                    anomaly_type TEXT NOT NULL,
                    anomaly_score REAL NOT NULL,
                    detection_method TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Calibration history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS calibration_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sensor_type TEXT NOT NULL,
                    calibration_type TEXT NOT NULL,
                    parameters TEXT NOT NULL,
                    applied_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    accuracy_metrics TEXT
                )
            ''')
            
            # Alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    sensor_type TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    acknowledged BOOLEAN DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Predictions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sensor_type TEXT NOT NULL,
                    prediction_method TEXT NOT NULL,
                    forecast_data TEXT NOT NULL,
                    forecast_days INTEGER NOT NULL,
                    confidence_score REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_sensor_timestamp ON sensor_readings(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_anomaly_timestamp ON anomalies(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alert_timestamp ON alerts(timestamp)')
            
            conn.commit()
    
    def store_sensor_data(self, sensor_data):
        """Store sensor reading in database"""
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    cursor.execute('''
                        INSERT INTO sensor_readings 
                        (timestamp, temperature, pressure, vibration, humidity, is_calibrated)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        sensor_data['timestamp'],
                        sensor_data.get('temperature'),
                        sensor_data.get('pressure'),
                        sensor_data.get('vibration'),
                        sensor_data.get('humidity'),
                        True  # Assuming data is calibrated
                    ))
                    
                    conn.commit()
                    return cursor.lastrowid
            
            except Exception as e:
                print(f"Error storing sensor data: {e}")
                return None
    
    def store_batch_sensor_data(self, data_df):
        """Store multiple sensor readings in batch"""
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    # Prepare data for insertion
                    data_df_copy = data_df.copy()
                    data_df_copy['is_calibrated'] = True
                    
                    # Insert data
                    data_df_copy.to_sql('sensor_readings', conn, if_exists='append', index=False)
                    conn.commit()
                    
                    return True
            
            except Exception as e:
                print(f"Error storing batch sensor data: {e}")
                return False
    
    def get_recent_data(self, limit=100, sensor_types=None):
        """Get recent sensor readings"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = 'SELECT * FROM sensor_readings ORDER BY timestamp DESC LIMIT ?'
                df = pd.read_sql_query(query, conn, params=(limit,))
                
                if not df.empty:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.sort_values('timestamp')
                
                return df
        
        except Exception as e:
            print(f"Error retrieving recent data: {e}")
            return pd.DataFrame()
    
    def get_data_by_date_range(self, start_date, end_date):
        """Get sensor data within date range"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = '''
                    SELECT * FROM sensor_readings 
                    WHERE timestamp BETWEEN ? AND ?
                    ORDER BY timestamp
                '''
                df = pd.read_sql_query(query, conn, params=(start_date, end_date))
                
                if not df.empty:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                return df
        
        except Exception as e:
            print(f"Error retrieving data by date range: {e}")
            return pd.DataFrame()
    
    def store_anomaly(self, anomaly_data):
        """Store detected anomaly"""
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    cursor.execute('''
                        INSERT INTO anomalies 
                        (timestamp, sensor_type, sensor_value, anomaly_type, 
                         anomaly_score, detection_method)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        anomaly_data['timestamp'],
                        anomaly_data['sensor_type'],
                        anomaly_data['sensor_value'],
                        anomaly_data['anomaly_type'],
                        anomaly_data['anomaly_score'],
                        anomaly_data['detection_method']
                    ))
                    
                    conn.commit()
                    return cursor.lastrowid
            
            except Exception as e:
                print(f"Error storing anomaly: {e}")
                return None
    
    def store_batch_anomalies(self, anomalies_df):
        """Store multiple anomalies in batch"""
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    # Prepare data for insertion
                    anomalies_data = []
                    for _, row in anomalies_df.iterrows():
                        # Find sensor type and value from row
                        sensor_type = None
                        sensor_value = None
                        
                        for col in ['temperature', 'pressure', 'vibration', 'humidity']:
                            if col in row and pd.notna(row[col]):
                                sensor_type = col
                                sensor_value = row[col]
                                break
                        
                        if sensor_type:
                            anomalies_data.append({
                                'timestamp': row['timestamp'],
                                'sensor_type': sensor_type,
                                'sensor_value': sensor_value,
                                'anomaly_type': row['anomaly_type'],
                                'anomaly_score': row['anomaly_score'],
                                'detection_method': row['method']
                            })
                    
                    if anomalies_data:
                        anomalies_batch_df = pd.DataFrame(anomalies_data)
                        anomalies_batch_df.to_sql('anomalies', conn, if_exists='append', index=False)
                        conn.commit()
                    
                    return True
            
            except Exception as e:
                print(f"Error storing batch anomalies: {e}")
                return False
    
    def get_anomalies(self, start_date=None, end_date=None, sensor_type=None):
        """Get anomalies with optional filtering"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = 'SELECT * FROM anomalies WHERE 1=1'
                params = []
                
                if start_date:
                    query += ' AND timestamp >= ?'
                    params.append(start_date)
                
                if end_date:
                    query += ' AND timestamp <= ?'
                    params.append(end_date)
                
                if sensor_type:
                    query += ' AND sensor_type = ?'
                    params.append(sensor_type)
                
                query += ' ORDER BY timestamp DESC'
                
                df = pd.read_sql_query(query, conn, params=params)
                
                if not df.empty:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                return df
        
        except Exception as e:
            print(f"Error retrieving anomalies: {e}")
            return pd.DataFrame()
    
    def store_calibration_history(self, sensor_type, calibration_type, parameters, accuracy_metrics=None):
        """Store calibration history"""
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    cursor.execute('''
                        INSERT INTO calibration_history 
                        (sensor_type, calibration_type, parameters, accuracy_metrics)
                        VALUES (?, ?, ?, ?)
                    ''', (
                        sensor_type,
                        calibration_type,
                        json.dumps(parameters),
                        json.dumps(accuracy_metrics) if accuracy_metrics else None
                    ))
                    
                    conn.commit()
                    return cursor.lastrowid
            
            except Exception as e:
                print(f"Error storing calibration history: {e}")
                return None
    
    def get_calibration_history(self, sensor_type=None, limit=50):
        """Get calibration history"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = 'SELECT * FROM calibration_history'
                params = []
                
                if sensor_type:
                    query += ' WHERE sensor_type = ?'
                    params.append(sensor_type)
                
                query += ' ORDER BY applied_at DESC LIMIT ?'
                params.append(limit)
                
                df = pd.read_sql_query(query, conn, params=params)
                
                if not df.empty:
                    df['applied_at'] = pd.to_datetime(df['applied_at'])
                    # Parse JSON parameters
                    df['parameters'] = df['parameters'].apply(lambda x: json.loads(x) if x else {})
                    df['accuracy_metrics'] = df['accuracy_metrics'].apply(
                        lambda x: json.loads(x) if x else {}
                    )
                
                return df
        
        except Exception as e:
            print(f"Error retrieving calibration history: {e}")
            return pd.DataFrame()
    
    def store_alert(self, timestamp, sensor_type, alert_type, message, severity):
        """Store alert"""
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    cursor.execute('''
                        INSERT INTO alerts 
                        (timestamp, sensor_type, alert_type, message, severity)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (timestamp, sensor_type, alert_type, message, severity))
                    
                    conn.commit()
                    return cursor.lastrowid
            
            except Exception as e:
                print(f"Error storing alert: {e}")
                return None
    
    def get_alerts(self, acknowledged=None, limit=100):
        """Get alerts"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = 'SELECT * FROM alerts'
                params = []
                
                if acknowledged is not None:
                    query += ' WHERE acknowledged = ?'
                    params.append(acknowledged)
                
                query += ' ORDER BY timestamp DESC LIMIT ?'
                params.append(limit)
                
                df = pd.read_sql_query(query, conn, params=params)
                
                if not df.empty:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df['created_at'] = pd.to_datetime(df['created_at'])
                
                return df
        
        except Exception as e:
            print(f"Error retrieving alerts: {e}")
            return pd.DataFrame()
    
    def acknowledge_alert(self, alert_id):
        """Acknowledge an alert"""
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    cursor.execute(
                        'UPDATE alerts SET acknowledged = 1 WHERE id = ?',
                        (alert_id,)
                    )
                    
                    conn.commit()
                    return cursor.rowcount > 0
            
            except Exception as e:
                print(f"Error acknowledging alert: {e}")
                return False
    
    def store_prediction(self, sensor_type, prediction_method, forecast_data, forecast_days, confidence_score=None):
        """Store prediction results"""
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    cursor.execute('''
                        INSERT INTO predictions 
                        (sensor_type, prediction_method, forecast_data, forecast_days, confidence_score)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (
                        sensor_type,
                        prediction_method,
                        json.dumps(forecast_data),
                        forecast_days,
                        confidence_score
                    ))
                    
                    conn.commit()
                    return cursor.lastrowid
            
            except Exception as e:
                print(f"Error storing prediction: {e}")
                return None
    
    def get_predictions(self, sensor_type=None, limit=50):
        """Get prediction history"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = 'SELECT * FROM predictions'
                params = []
                
                if sensor_type:
                    query += ' WHERE sensor_type = ?'
                    params.append(sensor_type)
                
                query += ' ORDER BY created_at DESC LIMIT ?'
                params.append(limit)
                
                df = pd.read_sql_query(query, conn, params=params)
                
                if not df.empty:
                    df['created_at'] = pd.to_datetime(df['created_at'])
                    df['forecast_data'] = df['forecast_data'].apply(
                        lambda x: json.loads(x) if x else []
                    )
                
                return df
        
        except Exception as e:
            print(f"Error retrieving predictions: {e}")
            return pd.DataFrame()
    
    def get_database_stats(self):
        """Get database statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                stats = {}
                
                # Count records in each table
                tables = ['sensor_readings', 'anomalies', 'calibration_history', 'alerts', 'predictions']
                for table in tables:
                    cursor.execute(f'SELECT COUNT(*) FROM {table}')
                    stats[f'{table}_count'] = cursor.fetchone()[0]
                
                # Get date range of sensor data
                cursor.execute('SELECT MIN(timestamp), MAX(timestamp) FROM sensor_readings')
                min_date, max_date = cursor.fetchone()
                stats['data_range'] = {'start': min_date, 'end': max_date}
                
                # Get database file size
                stats['db_size_mb'] = os.path.getsize(self.db_path) / (1024 * 1024) if os.path.exists(self.db_path) else 0
                
                return stats
        
        except Exception as e:
            print(f"Error getting database stats: {e}")
            return {}
    
    def cleanup_old_data(self, days_to_keep=30):
        """Clean up old data beyond specified days"""
        with self.lock:
            try:
                cutoff_date = datetime.now() - timedelta(days=days_to_keep)
                
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # Delete old sensor readings
                    cursor.execute('DELETE FROM sensor_readings WHERE timestamp < ?', (cutoff_date,))
                    sensor_deleted = cursor.rowcount
                    
                    # Delete old anomalies
                    cursor.execute('DELETE FROM anomalies WHERE timestamp < ?', (cutoff_date,))
                    anomaly_deleted = cursor.rowcount
                    
                    # Delete old alerts
                    cursor.execute('DELETE FROM alerts WHERE timestamp < ?', (cutoff_date,))
                    alert_deleted = cursor.rowcount
                    
                    conn.commit()
                    
                    return {
                        'sensor_readings_deleted': sensor_deleted,
                        'anomalies_deleted': anomaly_deleted,
                        'alerts_deleted': alert_deleted
                    }
            
            except Exception as e:
                print(f"Error cleaning up old data: {e}")
                return {}
    
    def export_data_to_csv(self, table_name, output_path, start_date=None, end_date=None):
        """Export table data to CSV"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = f'SELECT * FROM {table_name}'
                params = []
                
                if start_date and table_name in ['sensor_readings', 'anomalies', 'alerts']:
                    query += ' WHERE timestamp >= ?'
                    params.append(start_date)
                    
                    if end_date:
                        query += ' AND timestamp <= ?'
                        params.append(end_date)
                elif end_date and table_name in ['sensor_readings', 'anomalies', 'alerts']:
                    query += ' WHERE timestamp <= ?'
                    params.append(end_date)
                
                df = pd.read_sql_query(query, conn, params=params)
                df.to_csv(output_path, index=False)
                
                return True
        
        except Exception as e:
            print(f"Error exporting data to CSV: {e}")
            return False
