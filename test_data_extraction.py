import pytest
from src.data.extraction import WeatherDataExtractor
from unittest.mock import patch, Mock

def test_fetch_historical_data():
    extractor = WeatherDataExtractor()
    
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.json.return_value = {
            "daily": {
                "time": ["2023-01-01", "2023-01-02"],
                "temperature_2m_max": [15.0, 16.0],
                "temperature_2m_min": [5.0, 6.0],
                "precipitation_sum": [0.0, 0.2]
            }
        }
        mock_get.return_value = mock_response
        
        df = extractor.fetch_historical_data("2023-01-01", "2023-01-02")
        assert len(df) == 2
        assert 'temp_max' in df.columns
