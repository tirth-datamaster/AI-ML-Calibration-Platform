import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import plotly.colors as pc

class Visualizer:
    """Handles all visualization needs for the calibration platform"""
    
    def __init__(self):
        self.color_palette = {
            'temperature': '#FF6B6B',  # Red
            'pressure': '#4ECDC4',     # Teal
            'vibration': '#45B7D1',   # Blue
            'humidity': '#96CEB4',     # Green
            'anomaly': '#FF4757',      # Bright red
            'prediction': '#FFA502',   # Orange
            'normal': '#2ECC71',       # Green
            'warning': '#F39C12',      # Orange
            'critical': '#E74C3C'      # Red
        }
        
        self.default_layout = {
            'template': 'plotly_white',
            'font': {'family': 'Arial, sans-serif', 'size': 12},
            'margin': {'l': 60, 'r': 30, 't': 60, 'b': 60},
            'hovermode': 'x unified'
        }
    
    def create_sensor_trends(self, data_df: pd.DataFrame, 
                           height: int = 600) -> go.Figure:
        """Create multi-sensor trend visualization"""
        if data_df.empty:
            return self._create_empty_plot("No sensor data available")
        
        # Create subplots - 2x2 grid for 4 sensors
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Temperature (°C)', 'Pressure (hPa)', 
                           'Vibration (m/s²)', 'Humidity (%)'),
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        sensor_configs = [
            ('temperature', 1, 1, '°C'),
            ('pressure', 1, 2, 'hPa'),
            ('vibration', 2, 1, 'm/s²'),
            ('humidity', 2, 2, '%')
        ]
        
        for sensor, row, col, unit in sensor_configs:
            if sensor in data_df.columns:
                values = data_df[sensor].dropna()
                timestamps = pd.to_datetime(data_df['timestamp']).iloc[:len(values)]
                
                # Main trend line
                fig.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=values,
                        mode='lines',
                        name=sensor.title(),
                        line=dict(color=self.color_palette[sensor], width=2),
                        hovertemplate=f'<b>{sensor.title()}</b><br>' +
                                    'Time: %{x}<br>' +
                                    f'Value: %{{y:.2f}} {unit}<br>' +
                                    '<extra></extra>'
                    ),
                    row=row, col=col
                )
                
                # Add rolling average if enough data
                if len(values) > 10:
                    rolling_avg = values.rolling(window=min(10, len(values)//3), center=True).mean()
                    
                    fig.add_trace(
                        go.Scatter(
                            x=timestamps,
                            y=rolling_avg,
                            mode='lines',
                            name=f'{sensor.title()} Trend',
                            line=dict(color=self.color_palette[sensor], width=1, dash='dash'),
                            opacity=0.7,
                            hovertemplate=f'<b>{sensor.title()} Trend</b><br>' +
                                        'Time: %{x}<br>' +
                                        f'Avg: %{{y:.2f}} {unit}<br>' +
                                        '<extra></extra>'
                        ),
                        row=row, col=col
                    )
        
        # Update layout
        fig.update_layout(
            title='Real-time Sensor Monitoring Dashboard',
            height=height,
            showlegend=False,
            **self.default_layout
        )
        
        # Update x-axes
        for i in range(1, 5):
            fig.update_xaxes(title_text="Time", row=(i-1)//2 + 1, col=(i-1)%2 + 1)
        
        return fig
    
    def create_anomaly_plot(self, data_df: pd.DataFrame, anomalies_df: pd.DataFrame, 
                          sensor_type: str, height: int = 500) -> go.Figure:
        """Create anomaly detection visualization"""
        if data_df.empty:
            return self._create_empty_plot("No data available for anomaly analysis")
        
        if sensor_type not in data_df.columns:
            return self._create_empty_plot(f"Sensor {sensor_type} not found in data")
        
        fig = go.Figure()
        
        # Main sensor data
        values = data_df[sensor_type].dropna()
        timestamps = pd.to_datetime(data_df['timestamp']).iloc[:len(values)]
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=values,
            mode='lines',
            name=f'{sensor_type.title()} Normal',
            line=dict(color=self.color_palette[sensor_type], width=2),
            hovertemplate=f'<b>{sensor_type.title()}</b><br>' +
                         'Time: %{x}<br>' +
                         'Value: %{y:.2f}<br>' +
                         '<extra></extra>'
        ))
        
        # Add anomalies if any
        if not anomalies_df.empty:
            # Filter anomalies for this sensor
            sensor_anomalies = anomalies_df[
                anomalies_df.get('sensor_type', anomalies_df.columns[1] if len(anomalies_df.columns) > 1 else sensor_type) == sensor_type
            ] if 'sensor_type' in anomalies_df.columns else anomalies_df
            
            if not sensor_anomalies.empty:
                anomaly_timestamps = pd.to_datetime(sensor_anomalies['timestamp'])
                anomaly_values = sensor_anomalies[sensor_type] if sensor_type in sensor_anomalies.columns else sensor_anomalies.iloc[:, 1]
                
                # Different colors for different anomaly types
                anomaly_types = sensor_anomalies.get('anomaly_type', ['unknown'] * len(sensor_anomalies))
                
                unique_types = anomaly_types.unique() if hasattr(anomaly_types, 'unique') else ['unknown']
                colors = px.colors.qualitative.Set1[:len(unique_types)]
                
                for i, anom_type in enumerate(unique_types):
                    if hasattr(anomaly_types, 'isin'):
                        mask = anomaly_types == anom_type
                        type_timestamps = anomaly_timestamps[mask]
                        type_values = anomaly_values[mask]
                    else:
                        type_timestamps = anomaly_timestamps
                        type_values = anomaly_values
                    
                    fig.add_trace(go.Scatter(
                        x=type_timestamps,
                        y=type_values,
                        mode='markers',
                        name=f'Anomaly: {anom_type}',
                        marker=dict(
                            color=colors[i % len(colors)],
                            size=8,
                            symbol='x',
                            line=dict(width=2, color='white')
                        ),
                        hovertemplate=f'<b>Anomaly: {anom_type}</b><br>' +
                                     'Time: %{x}<br>' +
                                     'Value: %{y:.2f}<br>' +
                                     '<extra></extra>'
                    ))
        
        # Add statistical boundaries
        if len(values) > 10:
            mean_val = values.mean()
            std_val = values.std()
            
            # Add mean line
            fig.add_hline(
                y=mean_val,
                line_dash="dash",
                line_color="gray",
                annotation_text=f"Mean: {mean_val:.2f}",
                annotation_position="bottom right"
            )
            
            # Add ±2σ boundaries
            upper_bound = mean_val + 2 * std_val
            lower_bound = mean_val - 2 * std_val
            
            fig.add_hline(
                y=upper_bound,
                line_dash="dot",
                line_color="red",
                opacity=0.5,
                annotation_text=f"+2σ: {upper_bound:.2f}",
                annotation_position="top right"
            )
            
            fig.add_hline(
                y=lower_bound,
                line_dash="dot",
                line_color="red",
                opacity=0.5,
                annotation_text=f"-2σ: {lower_bound:.2f}",
                annotation_position="bottom right"
            )
        
        fig.update_layout(
            title=f'Anomaly Detection - {sensor_type.title()}',
            xaxis_title='Time',
            yaxis_title=f'{sensor_type.title()} Value',
            height=height,
            **self.default_layout
        )
        
        return fig
    
    def create_prediction_plot(self, historical_data: pd.DataFrame, 
                             prediction_result: Dict[str, Any], 
                             sensor_type: str, height: int = 500) -> go.Figure:
        """Create drift prediction visualization"""
        if historical_data.empty:
            return self._create_empty_plot("No historical data available")
        
        if not prediction_result or 'forecast' not in prediction_result:
            return self._create_empty_plot("No prediction data available")
        
        fig = go.Figure()
        
        # Historical data
        if sensor_type in historical_data.columns:
            hist_values = historical_data[sensor_type].dropna()
            hist_timestamps = pd.to_datetime(historical_data['timestamp']).iloc[:len(hist_values)]
            
            fig.add_trace(go.Scatter(
                x=hist_timestamps,
                y=hist_values,
                mode='lines',
                name='Historical Data',
                line=dict(color=self.color_palette[sensor_type], width=2),
                hovertemplate='<b>Historical</b><br>' +
                             'Time: %{x}<br>' +
                             'Value: %{y:.2f}<br>' +
                             '<extra></extra>'
            ))
        
        # Prediction data
        forecast_data = prediction_result['forecast']
        if forecast_data:
            forecast_df = pd.DataFrame(forecast_data)
            
            if 'ds' in forecast_df.columns and 'yhat' in forecast_df.columns:
                pred_timestamps = pd.to_datetime(forecast_df['ds'])
                pred_values = forecast_df['yhat']
                
                # Main prediction line
                fig.add_trace(go.Scatter(
                    x=pred_timestamps,
                    y=pred_values,
                    mode='lines',
                    name='Prediction',
                    line=dict(color=self.color_palette['prediction'], width=2, dash='dash'),
                    hovertemplate='<b>Prediction</b><br>' +
                                 'Time: %{x}<br>' +
                                 'Value: %{y:.2f}<br>' +
                                 '<extra></extra>'
                ))
                
                # Confidence intervals if available
                if 'yhat_lower' in forecast_df.columns and 'yhat_upper' in forecast_df.columns:
                    fig.add_trace(go.Scatter(
                        x=pred_timestamps,
                        y=forecast_df['yhat_upper'],
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=pred_timestamps,
                        y=forecast_df['yhat_lower'],
                        mode='lines',
                        fill='tonexty',
                        fillcolor=f'rgba{tuple(list(pc.hex_to_rgb(self.color_palette["prediction"])) + [0.2])}',
                        line=dict(width=0),
                        name='Confidence Interval',
                        hovertemplate='<b>Confidence Interval</b><br>' +
                                     'Time: %{x}<br>' +
                                     'Upper: %{y:.2f}<br>' +
                                     '<extra></extra>'
                    ))
        
        # Add vertical line separating historical and predicted data
        if not historical_data.empty:
            last_historical_time = pd.to_datetime(historical_data['timestamp']).max()
            fig.add_vline(
                x=last_historical_time,
                line_dash="solid",
                line_color="gray",
                annotation_text="Prediction Start",
                annotation_position="top"
            )
        
        # Add trend information
        trend_slope = prediction_result.get('trend_slope', 0)
        confidence = prediction_result.get('confidence', 0)
        
        fig.update_layout(
            title=f'Drift Prediction - {sensor_type.title()}<br>' +
                  f'<sub>Trend: {trend_slope:+.4f} units/day | Confidence: {confidence:.1f}%</sub>',
            xaxis_title='Time',
            yaxis_title=f'{sensor_type.title()} Value',
            height=height,
            **self.default_layout
        )
        
        return fig
    
    def create_correlation_heatmap(self, correlation_matrix: pd.DataFrame, 
                                 height: int = 400) -> go.Figure:
        """Create correlation heatmap between sensors"""
        if correlation_matrix.empty:
            return self._create_empty_plot("No correlation data available")
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 12},
            hovertemplate='<b>%{y} vs %{x}</b><br>' +
                         'Correlation: %{z:.3f}<br>' +
                         '<extra></extra>'
        ))
        
        fig.update_layout(
            title='Sensor Correlation Matrix',
            height=height,
            **self.default_layout
        )
        
        return fig
    
    def create_alert_timeline(self, alerts_data: List[Dict[str, Any]], 
                            height: int = 400) -> go.Figure:
        """Create timeline visualization of alerts"""
        if not alerts_data:
            return self._create_empty_plot("No alerts to display")
        
        fig = go.Figure()
        
        # Convert to DataFrame for easier processing
        alerts_df = pd.DataFrame(alerts_data)
        alerts_df['timestamp'] = pd.to_datetime(alerts_df['timestamp'])
        
        # Color mapping for severity levels
        severity_colors = {
            'critical': self.color_palette['critical'],
            'warning': self.color_palette['warning'],
            'info': '#3498DB'
        }
        
        # Group by severity
        for severity in alerts_df['level'].unique():
            severity_alerts = alerts_df[alerts_df['level'] == severity]
            
            fig.add_trace(go.Scatter(
                x=severity_alerts['timestamp'],
                y=severity_alerts['sensor_type'],
                mode='markers',
                name=severity.title(),
                marker=dict(
                    color=severity_colors.get(severity, '#95A5A6'),
                    size=10,
                    symbol='circle'
                ),
                text=severity_alerts['message'],
                hovertemplate='<b>%{text}</b><br>' +
                             'Time: %{x}<br>' +
                             'Sensor: %{y}<br>' +
                             f'Severity: {severity}<br>' +
                             '<extra></extra>'
            ))
        
        fig.update_layout(
            title='Alert Timeline',
            xaxis_title='Time',
            yaxis_title='Sensor Type',
            height=height,
            **self.default_layout
        )
        
        return fig
    
    def create_sensor_health_dashboard(self, health_scores: Dict[str, Dict[str, float]], 
                                     height: int = 400) -> go.Figure:
        """Create sensor health overview dashboard"""
        if not health_scores:
            return self._create_empty_plot("No health data available")
        
        # Create gauge charts for each sensor
        sensors = list(health_scores.keys())
        n_sensors = len(sensors)
        
        if n_sensors == 0:
            return self._create_empty_plot("No sensor health data")
        
        # Create subplots for gauges
        cols = 2
        rows = (n_sensors + 1) // 2
        
        fig = make_subplots(
            rows=rows,
            cols=cols,
            specs=[[{"type": "indicator"}] * cols for _ in range(rows)],
            subplot_titles=[sensor.title() for sensor in sensors]
        )
        
        for i, sensor in enumerate(sensors):
            row = i // cols + 1
            col = i % cols + 1
            
            score = health_scores[sensor].get('health_score', 0)
            
            # Determine color based on score
            if score >= 80:
                color = self.color_palette['normal']
            elif score >= 60:
                color = self.color_palette['warning']
            else:
                color = self.color_palette['critical']
            
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': f"{sensor.title()}<br>Health Score"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': color},
                        'steps': [
                            {'range': [0, 60], 'color': "lightgray"},
                            {'range': [60, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title='Sensor Health Dashboard',
            height=height,
            **self.default_layout
        )
        
        return fig
    
    def create_frequency_spectrum(self, frequency_analysis: Dict[str, Any], 
                                sensor_type: str, height: int = 400) -> go.Figure:
        """Create frequency spectrum visualization"""
        if not frequency_analysis or 'dominant_frequencies' not in frequency_analysis:
            return self._create_empty_plot("No frequency analysis data available")
        
        fig = go.Figure()
        
        dominant_freqs = frequency_analysis['dominant_frequencies']
        
        if dominant_freqs:
            frequencies = [f['frequency'] for f in dominant_freqs]
            magnitudes = [f['magnitude'] for f in dominant_freqs]
            
            fig.add_trace(go.Bar(
                x=frequencies,
                y=magnitudes,
                name='Frequency Components',
                marker_color=self.color_palette[sensor_type],
                hovertemplate='<b>Frequency Component</b><br>' +
                             'Frequency: %{x:.4f} Hz<br>' +
                             'Magnitude: %{y:.2f}<br>' +
                             'Period: %{customdata:.2f} sec<br>' +
                             '<extra></extra>',
                customdata=[f['period'] for f in dominant_freqs]
            ))
        
        fig.update_layout(
            title=f'Frequency Spectrum - {sensor_type.title()}',
            xaxis_title='Frequency (Hz)',
            yaxis_title='Magnitude',
            height=height,
            **self.default_layout
        )
        
        return fig
    
    def create_data_quality_report(self, quality_metrics: Dict[str, Dict[str, Any]], 
                                 height: int = 500) -> go.Figure:
        """Create data quality report visualization"""
        if not quality_metrics:
            return self._create_empty_plot("No quality metrics available")
        
        # Create subplots for different quality aspects
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Overall Quality Scores', 'Completeness', 
                           'Accuracy', 'Consistency'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        sensors = list(quality_metrics.keys())
        
        # Overall quality scores
        overall_scores = [quality_metrics[s].get('quality_score', 0) for s in sensors]
        fig.add_trace(
            go.Bar(x=sensors, y=overall_scores, name='Quality Score',
                   marker_color=[self._get_quality_color(score) for score in overall_scores]),
            row=1, col=1
        )
        
        # Completeness scores
        completeness = [quality_metrics[s].get('metrics', {}).get('completeness', 0) for s in sensors]
        fig.add_trace(
            go.Bar(x=sensors, y=completeness, name='Completeness',
                   marker_color=self.color_palette['temperature']),
            row=1, col=2
        )
        
        # Accuracy (inverse of outliers)
        accuracy = [quality_metrics[s].get('metrics', {}).get('outlier_free', 0) for s in sensors]
        fig.add_trace(
            go.Bar(x=sensors, y=accuracy, name='Accuracy',
                   marker_color=self.color_palette['pressure']),
            row=2, col=1
        )
        
        # Consistency
        consistency = [quality_metrics[s].get('metrics', {}).get('consistency', 0) for s in sensors]
        fig.add_trace(
            go.Bar(x=sensors, y=consistency, name='Consistency',
                   marker_color=self.color_palette['vibration']),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Data Quality Assessment Report',
            height=height,
            showlegend=False,
            **self.default_layout
        )
        
        # Update y-axes to show percentage
        for i in range(1, 5):
            row = (i-1) // 2 + 1
            col = (i-1) % 2 + 1
            fig.update_yaxes(title_text="Score (%)", range=[0, 100], row=row, col=col)
        
        return fig
    
    def _get_quality_color(self, score: float) -> str:
        """Get color based on quality score"""
        if score >= 90:
            return self.color_palette['normal']
        elif score >= 70:
            return self.color_palette['warning']
        else:
            return self.color_palette['critical']
    
    def _create_empty_plot(self, message: str) -> go.Figure:
        """Create an empty plot with a message"""
        fig = go.Figure()
        
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            xanchor='center', yanchor='middle',
            font=dict(size=16, color="gray"),
            showarrow=False
        )
        
        fig.update_layout(
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            **self.default_layout
        )
        
        return fig
    
    def create_calibration_drift_plot(self, calibration_history: pd.DataFrame, 
                                    sensor_type: str, height: int = 400) -> go.Figure:
        """Create calibration drift visualization over time"""
        if calibration_history.empty:
            return self._create_empty_plot("No calibration history available")
        
        fig = go.Figure()
        
        if 'applied_at' in calibration_history.columns:
            timestamps = pd.to_datetime(calibration_history['applied_at'])
            
            # Plot calibration offset changes over time
            if 'parameters' in calibration_history.columns:
                offsets = []
                slopes = []
                
                for _, row in calibration_history.iterrows():
                    params = row['parameters']
                    if isinstance(params, dict) and sensor_type in params:
                        sensor_params = params[sensor_type]
                        offsets.append(sensor_params.get('offset', 0))
                        slopes.append(sensor_params.get('slope', 1))
                    else:
                        offsets.append(0)
                        slopes.append(1)
                
                # Offset drift
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=offsets,
                    mode='lines+markers',
                    name='Calibration Offset',
                    line=dict(color=self.color_palette[sensor_type], width=2),
                    marker=dict(size=6),
                    yaxis='y',
                    hovertemplate='<b>Calibration Offset</b><br>' +
                                 'Time: %{x}<br>' +
                                 'Offset: %{y:.4f}<br>' +
                                 '<extra></extra>'
                ))
                
                # Slope drift
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=slopes,
                    mode='lines+markers',
                    name='Calibration Slope',
                    line=dict(color=self.color_palette['prediction'], width=2),
                    marker=dict(size=6),
                    yaxis='y2',
                    hovertemplate='<b>Calibration Slope</b><br>' +
                                 'Time: %{x}<br>' +
                                 'Slope: %{y:.4f}<br>' +
                                 '<extra></extra>'
                ))
        
        fig.update_layout(
            title=f'Calibration History - {sensor_type.title()}',
            xaxis_title='Time',
            yaxis=dict(title='Offset', side='left'),
            yaxis2=dict(title='Slope', side='right', overlaying='y'),
            height=height,
            **self.default_layout
        )
        
        return fig
