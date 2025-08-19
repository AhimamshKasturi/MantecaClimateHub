from db_config import get_sqlalchemy_url
from sqlalchemy import create_engine
import pandas as pd


engine = create_engine(get_sqlalchemy_url())

'''
select date, temperature, humidity, precipitation from manteca_data.daily_weather 
where date is not null order by date asc;
'''
query="""
select * 
from manteca_data.daily_weather
where date is not null 
order by date asc;    
"""

query="""
select date, avg_temp_f as temperature, avg_humidity_pct as humidity, precip_in as precipitation
from manteca_data.daily_weather
where date is not null 
order by date asc;    
"""

df=pd.read_sql(query,engine)
df.info()
df["date"]=pd.to_datetime(df["date"])

#lag features
#past values of a variable used as predictors for forecast models

df["temperature_lag_1"] = df["temperature"].shift(1)
df["humidity_lag_1"] = df["humidity"].shift(1)

#rolling averages
#method to find average of specific number of data points 
df["temperature_rolling_average"] = df["temperature"].rolling(window=7).mean()
df["humidity_rolling_average"] = df["humidity"].rolling(window=7).mean()

#binning
df["temperature_bin"]=pd.cut(df["temperature"],bins=[30, 50, 70, 90, 110], labels =["Cold", "Mild", "Warm", "Hot"])

#trend detection
df["temperature_trend"]=df["temperature"].diff().rolling(window=3).mean()

print(df.info())
print(df.head())

df.to_sql("daily_weather_enriched", engine, schema="manteca_data", if_exists="replace", index=False)
print("table created in pgadmin")
