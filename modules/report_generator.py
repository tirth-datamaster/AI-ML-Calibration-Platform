import pandas as pd
import numpy as np
import io
import base64
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

class ReportGenerator:
    """Generates various types of reports for the calibration platform"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.custom_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=20,
            textColor=colors.darkblue
        )
    
    def generate_summary_report(self, data_df):
        """Generate summary report with basic statistics"""
        if data_df.empty:
            return pd.DataFrame()
        
        summary_data = []
        
        # Calculate statistics for each sensor
        sensor_types = ['temperature', 'pressure', 'vibration', 'humidity']
        
        for sensor in sensor_types:
            if sensor in data_df.columns:
                values = data_df[sensor].dropna()
                
                if len(values) > 0:
                    summary_data.append({
                        'sensor_type': sensor.title(),
                        'count': len(values),
                        'mean': values.mean(),
                        'std': values.std(),
                        'min': values.min(),
                        'max': values.max(),
                        'range': values.max() - values.min(),
                        'latest_value': values.iloc[-1],
                        'data_quality': self._calculate_data_quality(values)
                    })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Add time-based statistics
        if 'timestamp' in data_df.columns:
            time_stats = self._calculate_time_statistics(data_df)
            summary_df = pd.concat([summary_df, pd.DataFrame([time_stats])], ignore_index=True)
        
        return summary_df
    
    def generate_anomaly_report(self, data_df, anomalies_df):
        """Generate anomaly analysis report"""
        if anomalies_df.empty:
            return pd.DataFrame({'message': ['No anomalies detected in the specified period.']})
        
        anomaly_summary = []
        
        # Group anomalies by type
        anomaly_by_type = anomalies_df.groupby('anomaly_type').agg({
            'anomaly_score': ['count', 'mean', 'max'],
            'timestamp': ['min', 'max']
        }).round(3)
        
        anomaly_by_type.columns = ['count', 'avg_score', 'max_score', 'first_occurrence', 'last_occurrence']
        anomaly_by_type = anomaly_by_type.reset_index()
        
        # Add severity classification
        anomaly_by_type['severity'] = anomaly_by_type['avg_score'].apply(self._classify_severity)
        
        # Calculate anomaly rate
        if not data_df.empty:
            total_readings = len(data_df)
            anomaly_rate = len(anomalies_df) / total_readings * 100
            
            anomaly_summary.append({
                'metric': 'Total Anomalies',
                'value': len(anomalies_df),
                'percentage': f"{anomaly_rate:.2f}%"
            })
        
        # Most problematic sensor
        if 'sensor_type' in anomalies_df.columns:
            sensor_anomaly_counts = anomalies_df.groupby('sensor_type').size()
            most_problematic = sensor_anomaly_counts.idxmax()
            
            anomaly_summary.append({
                'metric': 'Most Problematic Sensor',
                'value': most_problematic.title(),
                'percentage': f"{sensor_anomaly_counts[most_problematic]} anomalies"
            })
        
        # Combine results
        summary_df = pd.DataFrame(anomaly_summary)
        combined_df = pd.concat([
            summary_df,
            pd.DataFrame([{'metric': '--- Anomaly Types ---', 'value': '', 'percentage': ''}]),
            anomaly_by_type.rename(columns={'anomaly_type': 'metric', 'count': 'value', 'severity': 'percentage'})
        ], ignore_index=True)
        
        return combined_df
    
    def generate_calibration_report(self, data_df):
        """Generate calibration status report"""
        if data_df.empty:
            return pd.DataFrame({'message': ['No calibration data available.']})
        
        calibration_data = []
        sensor_types = ['temperature', 'pressure', 'vibration', 'humidity']
        
        for sensor in sensor_types:
            if sensor in data_df.columns:
                values = data_df[sensor].dropna()
                
                if len(values) > 0:
                    # Calculate drift over time
                    drift_analysis = self._analyze_sensor_drift(data_df, sensor)
                    
                    # Calculate stability metrics
                    stability = self._calculate_stability_metrics(values)
                    
                    calibration_data.append({
                        'sensor_type': sensor.title(),
                        'drift_rate': f"{drift_analysis['drift_rate']:.4f} units/hour",
                        'stability_score': f"{stability['stability_score']:.2f}%",
                        'calibration_accuracy': f"{stability['accuracy']:.2f}%",
                        'recommended_action': self._get_calibration_recommendation(
                            drift_analysis, stability
                        ),
                        'last_calibrated': 'Auto-calibrated',  # Placeholder
                        'next_calibration': self._estimate_next_calibration(drift_analysis)
                    })
        
        return pd.DataFrame(calibration_data)
    
    def export_to_excel(self, report_data):
        """Export report data to Excel format"""
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Write main report
            report_data.to_excel(writer, sheet_name='Report', index=False)
            
            # Format the worksheet
            worksheet = writer.sheets['Report']
            
            # Auto-adjust column widths
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width
        
        output.seek(0)
        return output.getvalue()
    
    def export_to_pdf(self, report_data, report_type):
        """Export report data to PDF format"""
        output = io.BytesIO()
        
        # Create PDF document
        doc = SimpleDocTemplate(output, pagesize=A4)
        story = []
        
        # Title
        title = Paragraph(f"{report_type} - {datetime.now().strftime('%Y-%m-%d %H:%M')}", 
                         self.custom_style)
        story.append(title)
        story.append(Spacer(1, 20))
        
        # Summary text
        summary_text = self._generate_report_summary_text(report_data, report_type)
        summary_para = Paragraph(summary_text, self.styles['Normal'])
        story.append(summary_para)
        story.append(Spacer(1, 20))
        
        # Data table
        if not report_data.empty:
            # Convert DataFrame to table data
            table_data = [report_data.columns.tolist()]
            for _, row in report_data.iterrows():
                table_data.append([str(val) for val in row.values])
            
            # Create table
            table = Table(table_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(table)
        
        # Add charts if possible
        chart_image = self._create_summary_chart(report_data, report_type)
        if chart_image:
            story.append(Spacer(1, 20))
            story.append(chart_image)
        
        # Build PDF
        doc.build(story)
        output.seek(0)
        return output.getvalue()
    
    def _calculate_data_quality(self, values):
        """Calculate data quality score based on various metrics"""
        if len(values) == 0:
            return 0.0
        
        # Check for missing values
        missing_penalty = 0
        
        # Check for outliers (values beyond 3 standard deviations)
        mean_val = values.mean()
        std_val = values.std()
        
        if std_val > 0:
            outliers = np.sum(np.abs(values - mean_val) > 3 * std_val)
            outlier_penalty = (outliers / len(values)) * 30
        else:
            outlier_penalty = 0
        
        # Check for stuck values (consecutive identical readings)
        stuck_penalty = 0
        if len(values) > 1:
            stuck_count = 0
            for i in range(1, len(values)):
                if abs(values.iloc[i] - values.iloc[i-1]) < 0.001:
                    stuck_count += 1
            stuck_penalty = (stuck_count / len(values)) * 20
        
        # Calculate final quality score
        quality_score = 100 - missing_penalty - outlier_penalty - stuck_penalty
        return max(0, min(100, quality_score))
    
    def _calculate_time_statistics(self, data_df):
        """Calculate time-based statistics"""
        timestamps = pd.to_datetime(data_df['timestamp'])
        
        time_stats = {
            'sensor_type': 'Time Analysis',
            'count': len(timestamps),
            'mean': 'N/A',
            'std': 'N/A',
            'min': timestamps.min().strftime('%Y-%m-%d %H:%M:%S'),
            'max': timestamps.max().strftime('%Y-%m-%d %H:%M:%S'),
            'range': str(timestamps.max() - timestamps.min()),
            'latest_value': 'N/A',
            'data_quality': self._calculate_temporal_quality(timestamps)
        }
        
        return time_stats
    
    def _calculate_temporal_quality(self, timestamps):
        """Calculate quality of temporal data"""
        if len(timestamps) < 2:
            return 100.0
        
        # Calculate time intervals
        intervals = timestamps.diff().dropna()
        
        # Expected interval (median)
        expected_interval = intervals.median()
        
        # Calculate deviation from expected interval
        deviations = np.abs(intervals - expected_interval)
        avg_deviation = deviations.mean()
        
        # Quality score based on consistency
        if expected_interval.total_seconds() > 0:
            consistency = 1 - (avg_deviation.total_seconds() / expected_interval.total_seconds())
            quality = max(0, min(100, consistency * 100))
        else:
            quality = 100.0
        
        return quality
    
    def _classify_severity(self, score):
        """Classify anomaly severity based on score"""
        if score >= 5.0:
            return 'Critical'
        elif score >= 3.0:
            return 'High'
        elif score >= 1.5:
            return 'Medium'
        else:
            return 'Low'
    
    def _analyze_sensor_drift(self, data_df, sensor_type):
        """Analyze sensor drift over time"""
        if sensor_type not in data_df.columns or len(data_df) < 10:
            return {'drift_rate': 0.0, 'direction': 'stable'}
        
        values = data_df[sensor_type].dropna()
        timestamps = pd.to_datetime(data_df['timestamp'])
        
        # Convert timestamps to hours from start
        start_time = timestamps.min()
        hours = [(t - start_time).total_seconds() / 3600 for t in timestamps[:len(values)]]
        
        # Calculate linear trend
        if len(hours) > 1:
            slope = np.polyfit(hours, values, 1)[0]
            
            direction = 'increasing' if slope > 0.001 else 'decreasing' if slope < -0.001 else 'stable'
            
            return {
                'drift_rate': abs(slope),
                'direction': direction,
                'slope': slope
            }
        
        return {'drift_rate': 0.0, 'direction': 'stable', 'slope': 0.0}
    
    def _calculate_stability_metrics(self, values):
        """Calculate sensor stability metrics"""
        if len(values) == 0:
            return {'stability_score': 0.0, 'accuracy': 0.0}
        
        # Coefficient of variation as stability measure
        mean_val = values.mean()
        std_val = values.std()
        
        if mean_val != 0:
            cv = (std_val / abs(mean_val)) * 100
            stability_score = max(0, 100 - cv * 5)  # Scale CV to stability score
        else:
            stability_score = 100.0
        
        # Accuracy based on consistency (inverse of variance)
        accuracy = max(0, 100 - std_val * 10)
        
        return {
            'stability_score': min(100, stability_score),
            'accuracy': min(100, accuracy)
        }
    
    def _get_calibration_recommendation(self, drift_analysis, stability):
        """Get calibration recommendation based on analysis"""
        drift_rate = drift_analysis['drift_rate']
        stability_score = stability['stability_score']
        
        if drift_rate > 0.1 or stability_score < 80:
            return 'Immediate Calibration Required'
        elif drift_rate > 0.05 or stability_score < 90:
            return 'Schedule Calibration Soon'
        else:
            return 'Normal Operation'
    
    def _estimate_next_calibration(self, drift_analysis):
        """Estimate when next calibration should occur"""
        drift_rate = drift_analysis['drift_rate']
        
        if drift_rate > 0.1:
            days = 1
        elif drift_rate > 0.05:
            days = 7
        elif drift_rate > 0.01:
            days = 30
        else:
            days = 90
        
        next_date = datetime.now() + timedelta(days=days)
        return next_date.strftime('%Y-%m-%d')
    
    def _generate_report_summary_text(self, report_data, report_type):
        """Generate summary text for PDF report"""
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        summary = f"""
        This {report_type.lower()} was generated on {current_time} by the AI/ML-Enhanced Calibration Platform.
        
        The report contains analysis of sensor data including statistical summaries, anomaly detection results,
        and calibration recommendations based on advanced machine learning algorithms.
        
        Key findings and recommendations are detailed in the table below.
        """
        
        if not report_data.empty and len(report_data) > 0:
            summary += f"\n\nTotal data points analyzed: {len(report_data)}"
        
        return summary
    
    def _create_summary_chart(self, report_data, report_type):
        """Create a summary chart for the report"""
        try:
            if report_data.empty:
                return None
            
            # Create a simple bar chart if numeric data is available
            numeric_columns = report_data.select_dtypes(include=[np.number]).columns
            
            if len(numeric_columns) > 0 and len(report_data) > 1:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Use first numeric column for chart
                y_col = numeric_columns[0]
                x_col = report_data.columns[0]  # First column as x-axis
                
                # Create bar chart
                ax.bar(range(len(report_data)), report_data[y_col])
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.set_title(f'{report_type} - {y_col} Overview')
                
                # Set x-axis labels
                ax.set_xticks(range(len(report_data)))
                ax.set_xticklabels(report_data[x_col], rotation=45, ha='right')
                
                plt.tight_layout()
                
                # Convert to image for PDF
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
                img_buffer.seek(0)
                
                # Create ReportLab image
                img = Image(img_buffer, width=6*inch, height=3.6*inch)
                plt.close()
                
                return img
        
        except Exception as e:
            print(f"Error creating chart: {e}")
        
        return None
    
    def generate_trend_analysis(self, data_df, sensor_type, days_back=30):
        """Generate detailed trend analysis for a specific sensor"""
        if sensor_type not in data_df.columns:
            return pd.DataFrame()
        
        # Filter recent data
        recent_data = data_df.tail(days_back * 24)  # Assume hourly data
        values = recent_data[sensor_type].dropna()
        
        if len(values) < 10:
            return pd.DataFrame({'message': ['Insufficient data for trend analysis']})
        
        # Calculate various trend metrics
        trend_metrics = []
        
        # Overall trend
        x = np.arange(len(values))
        slope, intercept = np.polyfit(x, values, 1)
        r_squared = np.corrcoef(x, values)[0, 1] ** 2
        
        trend_metrics.append({
            'metric': 'Overall Trend',
            'value': f"{'Increasing' if slope > 0 else 'Decreasing'} at {abs(slope):.4f} units/reading",
            'confidence': f"{r_squared:.3f}"
        })
        
        # Volatility
        volatility = values.std()
        mean_val = values.mean()
        cv = (volatility / mean_val * 100) if mean_val != 0 else 0
        
        trend_metrics.append({
            'metric': 'Volatility',
            'value': f"{volatility:.3f} (CV: {cv:.2f}%)",
            'confidence': 'N/A'
        })
        
        # Recent vs Historical
        if len(values) >= 20:
            recent_mean = values.tail(10).mean()
            historical_mean = values.head(10).mean()
            change_pct = ((recent_mean - historical_mean) / historical_mean * 100) if historical_mean != 0 else 0
            
            trend_metrics.append({
                'metric': 'Recent Change',
                'value': f"{change_pct:+.2f}% vs baseline",
                'confidence': f"Based on {len(values)} readings"
            })
        
        return pd.DataFrame(trend_metrics)
