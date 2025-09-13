# AI/ML-Enhanced Calibration Platform

## Overview

This is a real-time sensor monitoring platform built with Streamlit that provides intelligent anomaly detection, drift prediction, and automated calibration for industrial sensors. The system monitors temperature, pressure, vibration, and humidity sensors using AI/ML algorithms to detect anomalies, predict sensor drift, and maintain calibration accuracy. It features a comprehensive dashboard with real-time data visualization, alert management, and automated reporting capabilities.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application with wide layout configuration
- **Visualization**: Plotly for interactive charts and real-time data display
- **UI Components**: Multi-page dashboard with sidebar navigation and responsive design
- **State Management**: Streamlit session state for maintaining application state across interactions

### Backend Architecture
- **Core Modules**: Modular design with separate engines for calibration, anomaly detection, drift prediction, and alerting
- **Data Processing**: Real-time sensor simulation with configurable noise patterns and anomaly injection
- **ML Pipeline**: Multiple anomaly detection methods including Isolation Forest, statistical analysis, and pattern recognition
- **Prediction Engine**: Time series forecasting using Prophet, ARIMA, and linear trend analysis for drift prediction

### Data Storage Solutions
- **Primary Database**: SQLite for local data persistence
- **Schema Design**: Separate tables for sensor readings, anomalies, calibration history, alerts, and predictions
- **Data Management**: Thread-safe database operations with connection pooling and transaction management

### Authentication and Authorization
- **Current State**: No authentication system implemented
- **Access Control**: Open access to all platform features
- **Security Considerations**: Local deployment model without user management

### Real-time Processing
- **Sensor Simulation**: Configurable sensor simulator with realistic patterns, drift, and anomaly injection
- **Alert System**: Threshold-based alerting with configurable severity levels and notification channels
- **Calibration Engine**: Automated calibration with linear and polynomial correction algorithms
- **Data Pipeline**: Real-time data processing with noise filtering and smoothing capabilities

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework for the dashboard interface
- **Pandas/NumPy**: Data manipulation and numerical computing
- **Plotly**: Interactive visualization and charting
- **Scikit-learn**: Machine learning algorithms for anomaly detection
- **SciPy**: Scientific computing for signal processing and statistical analysis

### Time Series Forecasting
- **Prophet**: Facebook's time series forecasting library (optional dependency)
- **Statsmodels**: Statistical models for ARIMA and time series decomposition

### Database and Storage
- **SQLite3**: Local database for data persistence
- **Threading**: Multi-threaded database access with locking mechanisms

### Reporting and Documentation
- **ReportLab**: PDF report generation with charts and tables
- **Matplotlib/Seaborn**: Static plotting for report generation
- **Base64/IO**: Data encoding and in-memory file operations

### Data Processing
- **StandardScaler/MinMaxScaler**: Data normalization for ML algorithms
- **Signal Processing**: Scipy signal module for filtering and smoothing
- **Statistical Analysis**: Advanced statistical methods for anomaly detection

### Development and Utilities
- **Warnings**: Error handling and warning suppression
- **DateTime**: Time-based operations and scheduling
- **JSON**: Configuration and data serialization
- **OS**: File system operations and environment management