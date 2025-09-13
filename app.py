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
        page_title="Sensor Intelligence Platform",
        page_icon="🌟",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for aesthetic UI
    st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .metric-container {
        background: white;
        padding: 1.5rem;
        border-radius: 20px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        border: 1px solid #f0f2f6;
        margin: 0.5rem 0;
    }
    .status-good {
        background: linear-gradient(135deg, #4CAF50, #45a049);
        color: white;
        padding: 0.8rem;
        border-radius: 12px;
        text-align: center;
        font-weight: 600;
    }
    .status-warning {
        background: linear-gradient(135deg, #ff9800, #f57c00);
        color: white;
        padding: 0.8rem;
        border-radius: 12px;
        text-align: center;
        font-weight: 600;
    }
    .status-danger {
        background: linear-gradient(135deg, #f44336, #d32f2f);
        color: white;
        padding: 0.8rem;
        border-radius: 12px;
        text-align: center;
        font-weight: 600;
    }
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 25px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .sensor-card {
        background: white;
        padding: 1.5rem;
        border-radius: 18px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
        border-left: 5px solid #667eea;
        margin: 1rem 0;
        transition: transform 0.3s ease;
    }
    .sensor-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }
    .sidebar .stSelectbox > div > div {
        background: #f8f9fa;
        border-radius: 10px;
    }
    h1, h2, h3 {
        color: #2c3e50;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background: #f8f9fa;
        border-radius: 12px;
        border: 1px solid #e9ecef;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Hero section
    st.markdown("""
    <div class="hero-section">
        <h1 style="margin:0; color:white; font-size:2.5rem;">🌟 Sensor Intelligence Platform</h1>
        <p style="margin:0.5rem 0 0 0; color:rgba(255,255,255,0.9); font-size:1.2rem;">
            AI-powered sensor monitoring with real-time analytics
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Elegant sidebar
    st.sidebar.markdown("### 🎛️ Control Panel")
    
    # Simulation controls with better styling
    st.sidebar.markdown("#### 🔄 System Status")
    
    # Status indicator
    status_color = "🟢" if st.session_state.simulation_running else "🔴"
    status_text = "ACTIVE" if st.session_state.simulation_running else "INACTIVE"
    st.sidebar.markdown(f"{status_color} **{status_text}**")
    
    # Elegant button
    button_text = "⏸️ Stop Monitoring" if st.session_state.simulation_running else "▶️ Start Monitoring"
    if st.sidebar.button(button_text, use_container_width=True):
        st.session_state.simulation_running = not st.session_state.simulation_running
        if st.session_state.simulation_running:
            st.sidebar.success("✅ Monitoring started!")
        else:
            st.sidebar.info("⏸️ Monitoring paused!")
    
    st.sidebar.markdown("---")
    
    # Simplified threshold configuration
    st.sidebar.markdown("#### ⚠️ Alert Settings")
    
    # Quick preset buttons
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🛡️ Safe", use_container_width=True):
            st.session_state.alert_thresholds = {
                'temperature': {'min': 15, 'max': 40},
                'pressure': {'min': 950, 'max': 1050},
                'vibration': {'min': 0, 'max': 15},
                'humidity': {'min': 20, 'max': 90}
            }
    with col2:
        if st.button("⚡ Strict", use_container_width=True):
            st.session_state.alert_thresholds = {
                'temperature': {'min': 20, 'max': 30},
                'pressure': {'min': 990, 'max': 1020},
                'vibration': {'min': 0, 'max': 5},
                'humidity': {'min': 40, 'max': 70}
            }
    
    # Clean and modern tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Live Monitoring", 
        "🛡️ AI Detection", 
        "📈 Predictions", 
        "📋 Analytics"
    ])
    
    with tab1:
        show_realtime_dashboard()
    
    with tab2:
        show_anomaly_detection()
    
    with tab3:
        show_drift_prediction()
    
    with tab4:
        show_reports()
    
    # Auto-refresh for real-time updates
    if st.session_state.simulation_running:
        time.sleep(1)
        st.rerun()

def show_realtime_dashboard():
    # Clean header
    st.markdown("## 📊 Live Sensor Monitoring")
    
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
        # Get latest readings
        latest = recent_data.iloc[-1]
        
        # Beautiful sensor cards
        col1, col2 = st.columns(2)
        
        with col1:
            # Temperature card
            temp_status = "🟢 Normal" if st.session_state.alert_thresholds['temperature']['min'] <= latest['temperature'] <= st.session_state.alert_thresholds['temperature']['max'] else "🟠 Alert"
            st.markdown(f"""
            <div class="sensor-card">
                <h3 style="margin:0; color:#667eea;">🌡️ Temperature</h3>
                <h1 style="margin:0.5rem 0; color:#2c3e50; font-size:2.5rem;">{latest['temperature']:.1f}°C</h1>
                <p style="margin:0; color:#7f8c8d; font-size:1.1rem;">{temp_status}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Vibration card
            vib_status = "🟢 Normal" if st.session_state.alert_thresholds['vibration']['min'] <= latest['vibration'] <= st.session_state.alert_thresholds['vibration']['max'] else "🟠 Alert"
            st.markdown(f"""
            <div class="sensor-card">
                <h3 style="margin:0; color:#667eea;">📳 Vibration</h3>
                <h1 style="margin:0.5rem 0; color:#2c3e50; font-size:2.5rem;">{latest['vibration']:.1f}</h1>
                <p style="margin:0; color:#7f8c8d; font-size:1.1rem;">m/s² • {vib_status}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Pressure card
            press_status = "🟢 Normal" if st.session_state.alert_thresholds['pressure']['min'] <= latest['pressure'] <= st.session_state.alert_thresholds['pressure']['max'] else "🟠 Alert"
            st.markdown(f"""
            <div class="sensor-card">
                <h3 style="margin:0; color:#667eea;">🔘 Pressure</h3>
                <h1 style="margin:0.5rem 0; color:#2c3e50; font-size:2.5rem;">{latest['pressure']:.0f}</h1>
                <p style="margin:0; color:#7f8c8d; font-size:1.1rem;">hPa • {press_status}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Humidity card
            hum_status = "🟢 Normal" if st.session_state.alert_thresholds['humidity']['min'] <= latest['humidity'] <= st.session_state.alert_thresholds['humidity']['max'] else "🟠 Alert"
            st.markdown(f"""
            <div class="sensor-card">
                <h3 style="margin:0; color:#667eea;">💧 Humidity</h3>
                <h1 style="margin:0.5rem 0; color:#2c3e50; font-size:2.5rem;">{latest['humidity']:.0f}%</h1>
                <p style="margin:0; color:#7f8c8d; font-size:1.1rem;">{hum_status}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Simple trend chart
        st.markdown("### 📈 Live Trends")
        fig = st.session_state.visualizer.create_sensor_trends(recent_data)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown("""
        <div class="sensor-card" style="text-align:center; padding:3rem;">
            <h2 style="color:#667eea; margin:0;">🚀 Ready to Monitor</h2>
            <p style="color:#7f8c8d; margin:1rem 0 0 0; font-size:1.2rem;">
                Click "Start Monitoring" in the sidebar to begin live sensor tracking
            </p>
        </div>
        """, unsafe_allow_html=True)

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
