# 🌤️ Manteca Weather Analysis and Forecasting Project

## 📋 Project Overview

This project provides a comprehensive data pipeline for analyzing and forecasting weather patterns in Manteca, California. The system integrates data extraction, cleaning, feature engineering, machine learning modeling, and interactive visualization to deliver actionable insights into temperature trends and precipitation patterns.

## 🏗️ Project Structure

```
Manteca-Weather-Analysis/
├── 📊 Data Processing & ETL
│   ├── data_loader.py          # Data extraction from APIs
│   ├── data_cleaning.py        # Data cleaning and standardization
│   ├── feature_engineering.py  # Feature creation and transformation
│   └── data_uploader.py        # Database storage operations
│
├── 🤖 Machine Learning Models
│   ├── data_modelling.py       # Forecasting models (ARIMA, Prophet)
│   ├── arima_forecast.py       # SARIMAX forecasting implementation  [NEW]
│   ├── main.py                 # Main orchestration script
│   └── temperature_eda_level1.py # Exploratory data analysis
│
├── 🗄️ Database Management
│   ├── db_config.py           # Database configuration
│   ├── db_connection.py       # Connection handling
│   ├── table_creator.py       # Table schema management
│   └── schema_inspector.py    # Database schema inspection
│
├── 📈 Visualization & Analysis
│   ├── Temperature_EDA_Level1.ipynb  # Jupyter notebook for EDA
│   ├── EDA_Level_1.sql               # SQL queries for analysis
│   └── Manteca2023.xlsx              # Raw data reference
│
├── 📊 Tableau Dashboards
│   ├── Book1.twbx & variants         # Initial exploratory workbooks
│   ├── manteca_arima_prediction.twbx # ARIMA forecast visualizations
│   ├── manteca_project.twbx          # Main project dashboard
│   ├── manteca_project_Dashboard1.twbx # Primary dashboard view
│   └── manteca_project_Temperature Distribution.twbx # Temperature analysis
│
└── 📄 Documentation
    └── README.md              # This file
```

## 🚀 Key Features

### 🔍 Data Pipeline
- **Automated Data Extraction**: Fetches historical weather data from Open-Meteo API
- **Data Cleaning**: Handles missing values, outliers, and data standardization
- **Feature Engineering**: Creates derived features for improved modeling
- **Database Integration**: PostgreSQL storage with optimized schemas

### 📈 Forecasting Models
- **ARIMA/SARIMAX**: Time series forecasting with seasonal components
- **Facebook Prophet**: Advanced forecasting with trend and seasonality decomposition
- **Model Comparison**: Side-by-side evaluation of different forecasting approaches

### 📊 Visualization
- **Tableau Dashboards**: Interactive visualizations for:
  - Temperature trends and distributions
  - Forecast comparisons (ARIMA vs Prophet)
  - Seasonal pattern analysis
  - Prediction confidence intervals

### 🎯 Analytical Insights
- Temperature trend analysis for Manteca, California
- Seasonal pattern identification
- Forecast accuracy evaluation
- Data-driven weather predictions

## 🛠️ Technical Stack

- **Programming Language**: Python 3.10+
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Statsmodels, Prophet, pmdarima
- **Database**: PostgreSQL with SQLAlchemy
- **Visualization**: Tableau, Matplotlib, Seaborn
- **API Integration**: Requests library

## 📊 Data Sources

- **Primary**: Open-Meteo ERA5 API for historical weather data
- **Location**: Manteca, California (Latitude: 37.7974, Longitude: -121.2161)
- **Metrics**: Daily temperature (max/min), precipitation, derived features
- **Time Period**: 2023-01-01 to 2024-12-31 (historical), forecasts to 2026-12-31

## 🏃‍♂️ Getting Started

### Prerequisites
```bash
# Core dependencies
pip install pandas numpy sqlalchemy requests statsmodels matplotlib seaborn

# Optional: For Prophet forecasting
pip install prophet

# Optional: For auto-ARIMA functionality
pip install pmdarima
```

### Installation
1. Clone the repository
2. Set up PostgreSQL database
3. Configure database connection in `db_config.py`
4. Run the data pipeline: `python main.py`

### Database Setup
1. Create PostgreSQL database
2. Update `db_config.py` with your credentials:
```python
DB_CONFIG = {
    'host': 'localhost',
    'database': 'weather_db',
    'user': 'your_username',
    'password': 'your_password',
    'port': '5432'
}
```

### Typical Workflow
1. **Data Extraction**: `python data_loader.py`
2. **Data Cleaning**: `python data_cleaning.py`
3. **Feature Engineering**: `python feature_engineering.py`
4. **Model Training**: `python data_modelling.py` or `python arima_forecast.py`
5. **Visualization**: Open Tableau workbooks

## 🔮 Forecasting Modules

### ARIMA Forecasting (`arima_forecast.py`)
**Purpose**: Advanced time series forecasting using SARIMAX model
- Fetches historical weather data (2023-2024) from Open-Meteo ERA5 API
- Stores raw data in PostgreSQL under `manteca_data.arima_dataset`
- Implements SARIMAX with parameters (2,1,2)x(1,1,1,52)
- Weekly resampling for better seasonality capture
- Forecasts up to December 2026
- Exports multiple CSV formats for analysis

### Prophet Forecasting (`data_modelling.py`)
**Purpose**: Facebook Prophet implementation for comparison
- Handles daily seasonality and trends
- Provides confidence intervals
- Integrates with same data pipeline

## 📈 Output Files

### Database Tables
- `manteca_data.arima_dataset` - Historical weather data
- `manteca_data.arima_forecast` - ARIMA forecast results
- `manteca_data.temperature_forecast` - Prophet forecast results

### CSV Exports
- `arima_dataset.csv` - Historical temperature data
- `arima_forecast.csv` - Forecasted temperature values  
- `arima_combined.csv` - Combined actual and forecast data
- `model_comparison_*.csv` - Model performance comparisons

### Visualization Files
- Multiple Tableau workbooks (`.twbx`) for interactive analysis

## 🎨 Tableau Dashboards

The project includes multiple Tableau workbooks for different analytical perspectives:

1. **Main Dashboard** (`manteca_project.twbx`): Comprehensive overview
2. **Temperature Distribution** (`manteca_project_Temperature Distribution.twbx`): Statistical analysis
3. **ARIMA Predictions** (`manteca_arima_prediction.twbx`): Forecast visualizations
4. **Interactive Analysis** (`Book1*.twbx`): Exploratory workbooks

## 🔮 Forecasting Capabilities

- **Short-term**: 7-30 day temperature predictions
- **Medium-term**: Seasonal trend forecasting  
- **Long-term**: Multi-year climate trend analysis (up to 2026)
- **Model Comparison**: ARIMA vs Prophet performance evaluation
- **Confidence Intervals**: Prediction uncertainty quantification

## 📊 Sample Insights

- Manteca's temperature patterns show strong seasonal cycles
- ARIMA models capture short-term fluctuations effectively
- Prophet models excel at identifying yearly seasonality
- Both models provide complementary forecasting perspectives
- Weekly resampling improves seasonality capture for long-term forecasts

## 🚦 Usage Examples

### Run ARIMA Forecasting
```bash
python arima_forecast.py
```

### Run Full Pipeline
```bash
python main.py
```

### Generate Model Comparisons
```bash
python model_comparison.py
```

### Environment Variables
```bash
export OUTPUT_DIR_FORECAST="./forecast_results"
export DB_HOST="localhost"
export DB_NAME="weather_db"
```

## 🤝 Contributing

This project is open for extensions and improvements. Key areas for development:

- **Additional Parameters**: Humidity, wind speed, pressure data
- **Advanced Models**: LSTM, XGBoost for time series
- **Real-time Integration**: Live data feeds and alerts  
- **Geospatial Analysis**: Regional weather pattern comparisons
- **API Development**: REST endpoints for forecast access

## 🐛 Troubleshooting

### Common Issues
1. **Database Connection**: Verify credentials in `db_config.py`
2. **Missing Dependencies**: Run `pip install -r requirements.txt`
3. **API Limits**: Open-Meteo may have rate limits
4. **Memory Issues**: Large forecasts may require optimized processing

### Performance Tips
- Use weekly resampling for long-term forecasts
- Enable database connection pooling
- Cache API responses where possible
- Use chunk processing for large datasets

## 📊 Performance Metrics

The project includes built-in evaluation metrics:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE) 
- Mean Absolute Percentage Error (MAPE)
- Model comparison statistics

## 📄 License

This project is developed for educational and analytical purposes. Data sourced from Open-Meteo API under their terms of service.

## 👨‍💻 Author

**Ahimamsh Kasturi**  
- Weather Data Analysis & Forecasting
- Data Pipeline Development  
- Visualization & Dashboard Creation
- Machine Learning Implementation

---

*For detailed analysis and interactive visualizations, explore the Tableau workbooks included in this repository. The forecasting modules provide both short-term predictions and long-term climate trend analysis for Manteca, California.*
