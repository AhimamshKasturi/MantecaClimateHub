import requests
import pandas as pd
from typing import Dict, Any
import yaml

class WeatherDataExtractor:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
    def fetch_historical_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        params = self.config['api']['open_meteo']['params'].copy()
        params.update({
            "start_date": start_date,
            "end_date": end_date,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum"
        })
        
        response = requests.get(
            self.config['api']['open_meteo']['url'],
            params=params,
            timeout=self.config['api'].get('timeout', 30)
        )
        response.raise_for_status()
        
        return self._process_response(response.json())
    
    def _process_response(self, data: Dict) -> pd.DataFrame:
        df = pd.DataFrame(data["daily"])
        df["date"] = pd.to_datetime(df["time"])
        return df.rename(columns={
            "temperature_2m_max": "temp_max",
            "temperature_2m_min": "temp_min",
            "precipitation_sum": "precipitation"
        })[['date', 'temp_max', 'temp_min', 'precipitation']]
