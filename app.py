import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import os

# Import custom modules
from modules.sensor_simulator import SensorSimulator
from modules.calibration_engine import CalibrationEngine
from modules.anomaly_detector import AnomalyDetector
from modules.drift_predictor import DriftPredictor
from modules.database_manager import DatabaseManager
from modules.report_generator import ReportGenerator
from modules.alert_system import AlertSystem
from utils.data_processor import DataProcessor
from utils.visualization import Visualizer

# Initialize session state
if 'db_manager' not in st.session_state:
    st.session_state.db_manager = DatabaseManager()
    st.session_state.sensor_sim = SensorSimulator()
    st.session_state.calibration_engine = CalibrationEngine()
    st.session_state.anomaly_detector = AnomalyDetector()
    st.session_state.drift_predictor = DriftPredictor()
    st.session_state.alert_system = AlertSystem()
    st.session_state.report_generator = ReportGenerator()
    st.session_state.visualizer = Visualizer()
    st.session_state.data_processor = DataProcessor()
    st.session_state.simulation_running = False
    st.session_state.alert_thresholds = {
        'temperature': {'min': 18, 'max': 35},
        'pressure': {'min': 980, 'max': 1040},
        'vibration': {'min': 0, 'max': 10},
        'humidity': {'min': 30, 'max': 80}
    }

def main():
    st.set_page_config(
        page_title="AI/ML Calibration Platform",
        page_icon="🔧",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🔧 AI/ML-Enhanced Calibration Platform")
    st.markdown("Real-time sensor monitoring with intelligent anomaly detection and drift prediction")
    
    # Sidebar controls
    st.sidebar.header("Control Panel")
    
    # Simulation controls
    st.sidebar.subheader("Simulation Controls")
    if st.sidebar.button("Start Simulation" if not st.session_state.simulation_running else "Stop Simulation"):
        st.session_state.simulation_running = not st.session_state.simulation_running
        if st.session_state.simulation_running:
            st.sidebar.success("Simulation started!")
        else:
            st.sidebar.info("Simulation stopped!")
    
    # Threshold configuration
    st.sidebar.subheader("Alert Thresholds")
    for sensor_type in st.session_state.alert_thresholds.keys():
        st.sidebar.write(f"**{sensor_type.title()}**")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.session_state.alert_thresholds[sensor_type]['min'] = st.number_input(
                f"Min {sensor_type}", 
                value=st.session_state.alert_thresholds[sensor_type]['min'],
                key=f"min_{sensor_type}"
            )
        with col2:
            st.session_state.alert_thresholds[sensor_type]['max'] = st.number_input(
                f"Max {sensor_type}", 
                value=st.session_state.alert_thresholds[sensor_type]['max'],
                key=f"max_{sensor_type}"
            )
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Real-time Dashboard", 
        "🔍 Anomaly Detection", 
        "📈 Drift Prediction", 
        "📋 Reports", 
        "⚙️ Calibration"
    ])
    
    with tab1:
        show_realtime_dashboard()
    
    with tab2:
        show_anomaly_detection()
    
    with tab3:
        show_drift_prediction()
    
    with tab4:
        show_reports()
    
    with tab5:
        show_calibration()
    
    # Auto-refresh for real-time updates
    if st.session_state.simulation_running:
        time.sleep(1)
        st.rerun()

def show_realtime_dashboard():
    st.header("Real-time Sensor Dashboard")
    
    # Generate new sensor data if simulation is running
    if st.session_state.simulation_running:
        sensor_data = st.session_state.sensor_sim.generate_realtime_data()
        
        # Apply calibration
        calibrated_data = st.session_state.calibration_engine.apply_calibration(sensor_data)
        
        # Store in database
        st.session_state.db_manager.store_sensor_data(calibrated_data)
        
        # Check for alerts
        alerts = st.session_state.alert_system.check_thresholds(
            calibrated_data, st.session_state.alert_thresholds
        )
        
        # Display alerts
        if alerts:
            for alert in alerts:
                if alert['level'] == 'critical':
                    st.error(f"🚨 CRITICAL: {alert['message']}")
                elif alert['level'] == 'warning':
                    st.warning(f"⚠️ WARNING: {alert['message']}")
    
    # Get recent data for display
    recent_data = st.session_state.db_manager.get_recent_data(limit=100)
    
    if not recent_data.empty:
        # Current readings display
        st.subheader("Current Sensor Readings")
        
        # Get latest readings
        latest = recent_data.iloc[-1]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            temp_status = get_status_color(
                latest['temperature'], 
                st.session_state.alert_thresholds['temperature']
            )
            st.metric(
                "🌡️ Temperature", 
                f"{latest['temperature']:.2f}°C",
                delta=f"{latest['temperature'] - recent_data['temperature'].iloc[-2]:.2f}" if len(recent_data) > 1 else None
            )
            st.markdown(f"Status: {temp_status}")
        
        with col2:
            pressure_status = get_status_color(
                latest['pressure'], 
                st.session_state.alert_thresholds['pressure']
            )
            st.metric(
                "🔘 Pressure", 
                f"{latest['pressure']:.2f} hPa",
                delta=f"{latest['pressure'] - recent_data['pressure'].iloc[-2]:.2f}" if len(recent_data) > 1 else None
            )
            st.markdown(f"Status: {pressure_status}")
        
        with col3:
            vibration_status = get_status_color(
                latest['vibration'], 
                st.session_state.alert_thresholds['vibration']
            )
            st.metric(
                "📳 Vibration", 
                f"{latest['vibration']:.2f} m/s²",
                delta=f"{latest['vibration'] - recent_data['vibration'].iloc[-2]:.2f}" if len(recent_data) > 1 else None
            )
            st.markdown(f"Status: {vibration_status}")
        
        with col4:
            humidity_status = get_status_color(
                latest['humidity'], 
                st.session_state.alert_thresholds['humidity']
            )
            st.metric(
                "💧 Humidity", 
                f"{latest['humidity']:.2f}%",
                delta=f"{latest['humidity'] - recent_data['humidity'].iloc[-2]:.2f}" if len(recent_data) > 1 else None
            )
            st.markdown(f"Status: {humidity_status}")
        
        # Time series plots
        st.subheader("Sensor Trends")
        fig = st.session_state.visualizer.create_sensor_trends(recent_data)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No sensor data available. Start the simulation to begin monitoring.")

def show_anomaly_detection():
    st.header("Anomaly Detection Analysis")
    
    # Get data for analysis
    data = st.session_state.db_manager.get_recent_data(limit=1000)
    
    if not data.empty:
        # Anomaly detection controls
        col1, col2 = st.columns(2)
        with col1:
            sensor_type = st.selectbox("Select Sensor", ['temperature', 'pressure', 'vibration', 'humidity'])
        with col2:
            detection_method = st.selectbox("Detection Method", ['Isolation Forest', 'Statistical', 'Combined'])
        
        if st.button("Run Anomaly Detection"):
            # Detect anomalies
            anomalies = st.session_state.anomaly_detector.detect_anomalies(
                data, sensor_type, method=detection_method.lower().replace(' ', '_')
            )
            
            # Display results
            st.subheader(f"Anomaly Detection Results for {sensor_type.title()}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Data Points", len(data))
            with col2:
                st.metric("Anomalies Detected", len(anomalies))
            
            # Visualization
            fig = st.session_state.visualizer.create_anomaly_plot(data, anomalies, sensor_type)
            st.plotly_chart(fig, use_container_width=True)
            
            # Anomaly details
            if not anomalies.empty:
                st.subheader("Anomaly Details")
                st.dataframe(anomalies[['timestamp', sensor_type, 'anomaly_score', 'anomaly_type']])
    else:
        st.info("No data available for anomaly detection. Start the simulation to collect data.")

def show_drift_prediction():
    st.header("Sensor Drift Prediction")
    
    # Get data for prediction
    data = st.session_state.db_manager.get_recent_data(limit=1000)
    
    if len(data) >= 50:  # Need minimum data for prediction
        col1, col2 = st.columns(2)
        with col1:
            sensor_type = st.selectbox("Select Sensor for Prediction", ['temperature', 'pressure', 'vibration', 'humidity'])
        with col2:
            forecast_days = st.number_input("Forecast Days", min_value=1, max_value=30, value=7)
        
        prediction_method = st.selectbox("Prediction Method", ['Prophet', 'ARIMA', 'Linear Trend'])
        
        if st.button("Generate Prediction"):
            # Generate prediction
            prediction = st.session_state.drift_predictor.predict_drift(
                data, sensor_type, days=forecast_days, method=prediction_method.lower()
            )
            
            if prediction is not None:
                st.subheader(f"Drift Prediction for {sensor_type.title()}")
                
                # Create prediction visualization
                fig = st.session_state.visualizer.create_prediction_plot(
                    data, prediction, sensor_type
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Prediction statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Predicted Drift", f"{prediction['trend_slope']:.4f} units/day")
                with col2:
                    st.metric("Confidence Level", f"{prediction['confidence']:.1f}%")
                with col3:
                    st.metric("Forecast Horizon", f"{forecast_days} days")
            else:
                st.error("Unable to generate prediction. Insufficient data or model convergence issues.")
    else:
        st.info(f"Need at least 50 data points for drift prediction. Current: {len(data)}")

def show_reports():
    st.header("Reports and Data Export")
    
    # Report configuration
    col1, col2 = st.columns(2)
    with col1:
        report_type = st.selectbox("Report Type", ['Summary Report', 'Anomaly Report', 'Calibration Report'])
    with col2:
        export_format = st.selectbox("Export Format", ['PDF', 'Excel', 'CSV'])
    
    date_range = st.date_input(
        "Select Date Range",
        value=[datetime.now().date() - timedelta(days=7), datetime.now().date()],
        max_value=datetime.now().date()
    )
    
    if st.button("Generate Report"):
        # Get data for the specified date range
        start_date = datetime.combine(date_range[0], datetime.min.time()) if len(date_range) > 0 else datetime.now() - timedelta(days=7)
        end_date = datetime.combine(date_range[1], datetime.max.time()) if len(date_range) > 1 else datetime.now()
        
        data = st.session_state.db_manager.get_data_by_date_range(start_date, end_date)
        
        if not data.empty:
            # Generate report based on type
            if report_type == 'Summary Report':
                report_data = st.session_state.report_generator.generate_summary_report(data)
            elif report_type == 'Anomaly Report':
                anomalies = st.session_state.anomaly_detector.detect_anomalies(data, 'temperature')
                report_data = st.session_state.report_generator.generate_anomaly_report(data, anomalies)
            else:  # Calibration Report
                report_data = st.session_state.report_generator.generate_calibration_report(data)
            
            # Display report preview
            st.subheader("Report Preview")
            st.dataframe(report_data)
            
            # Export functionality
            if export_format == 'CSV':
                csv_data = report_data.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"{report_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            elif export_format == 'Excel':
                excel_data = st.session_state.report_generator.export_to_excel(report_data)
                st.download_button(
                    label="Download Excel",
                    data=excel_data,
                    file_name=f"{report_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:  # PDF
                pdf_data = st.session_state.report_generator.export_to_pdf(report_data, report_type)
                st.download_button(
                    label="Download PDF",
                    data=pdf_data,
                    file_name=f"{report_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )
        else:
            st.warning("No data available for the selected date range.")

def show_calibration():
    st.header("Calibration Engine Configuration")
    
    # Calibration parameters
    st.subheader("Calibration Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Temperature Calibration**")
        temp_offset = st.number_input("Temperature Offset (°C)", value=0.0, step=0.1)
        temp_slope = st.number_input("Temperature Slope", value=1.0, step=0.01)
        
        st.write("**Pressure Calibration**")
        pressure_offset = st.number_input("Pressure Offset (hPa)", value=0.0, step=0.1)
        pressure_slope = st.number_input("Pressure Slope", value=1.0, step=0.01)
    
    with col2:
        st.write("**Vibration Calibration**")
        vibration_offset = st.number_input("Vibration Offset (m/s²)", value=0.0, step=0.01)
        vibration_slope = st.number_input("Vibration Slope", value=1.0, step=0.01)
        
        st.write("**Humidity Calibration**")
        humidity_offset = st.number_input("Humidity Offset (%)", value=0.0, step=0.1)
        humidity_slope = st.number_input("Humidity Slope", value=1.0, step=0.01)
    
    # Filter settings
    st.subheader("Noise Filter Settings")
    filter_type = st.selectbox("Filter Type", ['Moving Average', 'Savitzky-Golay', 'Butterworth'])
    filter_window = st.slider("Filter Window Size", min_value=3, max_value=21, value=5, step=2)
    
    if st.button("Update Calibration Settings"):
        # Update calibration parameters
        calibration_params = {
            'temperature': {'offset': temp_offset, 'slope': temp_slope},
            'pressure': {'offset': pressure_offset, 'slope': pressure_slope},
            'vibration': {'offset': vibration_offset, 'slope': vibration_slope},
            'humidity': {'offset': humidity_offset, 'slope': humidity_slope}
        }
        
        filter_params = {
            'type': filter_type.lower().replace(' ', '_'),
            'window': filter_window
        }
        
        st.session_state.calibration_engine.update_parameters(calibration_params, filter_params)
        st.success("Calibration settings updated successfully!")
    
    # Show current calibration status
    st.subheader("Calibration Status")
    recent_data = st.session_state.db_manager.get_recent_data(limit=100)
    
    if not recent_data.empty:
        # Calculate calibration statistics
        stats = st.session_state.data_processor.calculate_calibration_stats(recent_data)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            temp_accuracy = stats.get('temperature', {}).get('accuracy', 0)
            st.metric("Temperature Accuracy", f"{temp_accuracy:.2f}%")
        with col2:
            pressure_accuracy = stats.get('pressure', {}).get('accuracy', 0)
            st.metric("Pressure Accuracy", f"{pressure_accuracy:.2f}%")
        with col3:
            vibration_accuracy = stats.get('vibration', {}).get('accuracy', 0)
            st.metric("Vibration Accuracy", f"{vibration_accuracy:.2f}%")
        with col4:
            humidity_accuracy = stats.get('humidity', {}).get('accuracy', 0)
            st.metric("Humidity Accuracy", f"{humidity_accuracy:.2f}%")
    else:
        st.info("No calibration data available. Start the simulation to begin collecting data.")

def get_status_color(value, thresholds):
    """Get color-coded status based on thresholds"""
    if value < thresholds['min'] or value > thresholds['max']:
        return "🔴 **Critical**"
    elif value < thresholds['min'] * 1.1 or value > thresholds['max'] * 0.9:
        return "🟡 **Warning**"
    else:
        return "🟢 **Normal**"

if __name__ == "__main__":
    main()
