# AI-ML-Calibration-Platform


This project is a Minimum Viable Product (MVP) for an AI/ML-Enhanced Calibration Platform built with Python, Streamlit, and SQLite. It simulates sensor readings, performs anomaly detection, and provides a simple dashboard for monitoring calibration data. The design is lightweight, modular, and ready for future upgrades.

🚀 Features

User Authentication: Fast login system with credentials stored in SQLite.
Dashboard: Displays real-time simulated sensor readings (temperature, pressure, humidity, vibration).
Charts & Trends: Interactive visualizations using Plotly.
Anomaly Detection: Detect anomalies using Isolation Forest or statistical thresholds.
Data Storage: All readings are stored in a local SQLite database.
Export: Export readings and anomalies to CSV.
Simple & Modular: Ready for future features like drift prediction, adaptive calibration, and alert systems.

🛠️ Tech Stack
Frontend / Dashboard: Streamlit
Backend / Logic: Python (Pandas, NumPy, Scikit-learn)
Database: SQLite (no PostgreSQL required)
Visualization: Plotly, Matplotlib



📂 Project Structure
📦 AI-ML-Calibration-Platform
 ┣ 📜 app.py              # Main Streamlit app (login, dashboard, charts, anomalies)
 ┣ 📜 database.py         # SQLite database setup and user credentials
 ┣ 📜 simulate_sensors.py # Sensor simulation and data insertion
 ┣ 📜 requirements.txt    # Project dependencies
 ┗ 📜 README.md           # Documentation


🔧 Installation & Setup
Clone the repository
git clone https://github.com/your-username/AI-ML-Calibration-Platform.git
cd AI-ML-Calibration-Platform


Create virtual environment (optional but recommended)

python -m venv venv
venv\Scripts\activate    # On Windows
source venv/bin/activate # On Mac/Linux


Install dependencies
pip install -r requirements.txt
Run the application
streamlit run app.py



Login credentials
Username: admin
Password: admin123


📊 Usage
On login, the dashboard shows latest sensor readings.
Click the Detect Anomalies checkbox to run anomaly detection.
View interactive charts for each sensor.
Export data to CSV for reporting or analysis.


🔮 Future Enhancements
Color-coded alerts (Green = Normal, Yellow = Warning, Red = Critical)
Drift prediction with ARIMA/LSTM models
Adaptive calibration algorithms
PDF/Excel export with charts
Real sensor integration
