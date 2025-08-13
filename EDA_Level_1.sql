SELECT * FROM manteca_data.daily_weather;

--inspecting tables structure
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_schema = 'manteca_data' AND table_name = 'daily_weather';

--checking range of date
SELECT MIN(date) AS start_date, MAX(date) AS end_date
FROM manteca_data.daily_weather;

--check row count
SELECT COUNT(*) AS total_rows
FROM manteca_data.daily_weather;

--check missing data
SELECT COUNT(*) AS missing_dates
FROM manteca_data.daily_weather
WHERE date IS NULL;

--check missing values in colunms
SELECT 
  COUNT(*) FILTER (WHERE avg_temp_f IS NULL) AS missing_temp,
  COUNT(*) FILTER (WHERE avg_humidity_pct IS NULL) AS missing_humidity,
  COUNT(*) FILTER (WHERE avg_wind_mph IS NULL) AS missing_wind,
  COUNT(*) FILTER (WHERE avg_pressure_in IS NULL) AS missing_pressure
FROM manteca_data.daily_weather;


--calculating average per month
SELECT 
  EXTRACT(MONTH FROM date) AS month,
  ROUND(AVG(avg_temp_f)::numeric, 2) AS avg_temp,
  ROUND(MAX(avg_temp_f)::numeric, 2) AS max_temp,
  ROUND(MIN(avg_temp_f)::numeric, 2) AS min_temp,
  ROUND(SUM(precip_in)::numeric, 2) AS total_precip,
  ROUND(AVG(avg_humidity_pct)::numeric, 2) AS avg_humidity,
  ROUND(AVG(avg_wind_mph)::numeric, 2) AS avg_wind,
  ROUND(AVG(avg_pressure_in)::numeric, 2) AS avg_pressure
FROM manteca_data.daily_weather
GROUP BY EXTRACT(MONTH FROM date)
ORDER BY month;




