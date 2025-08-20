import pandas as pd
from sqlalchemy import create_engine
from db_config import get_sqlalchemy_url
from sklearn.preprocessing import StandardScaler

engine = create_engine(get_sqlalchemy_url())

query="""
SELECT * FROM manteca_data.daily_weather_enriched
where date is not null 
order by date asc;    
"""
df=pd.read_sql(query,engine)
df.info()

#missing values
df = df.dropna(subset=["temperature_rolling_average", "humidity_rolling_average"])
print("Null points in temperature and humidity dropped successfully")
df["temperature_lag_1"]=df["temperature_lag_1"].fillna(df["temperature"])
df["humidity_lag_1"]=df["humidity_lag_1"].fillna(df["humidity"])
df["temperature_trend"]=df["temperature_trend"].fillna(0)
df["temperature_bin"] = df["temperature_bin"].fillna("Unknown")
print("missing values handled")

#date parsing and indexing
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)
print("checked date time format and modified as per our requirement")

#duplicate rows
df = df.drop_duplicates(subset=["date"])
print("duplicate rows removed")

#validate data types
print(df.dtypes)

#feature scaling : to standardize the range of independent variables or features 
scaler=StandardScaler()
df[["temperature","humidity","precipitation"]]=scaler.fit_transform(df[["temperature","humidity","precipitation"]])
#we are doing this to transform the data. mean becomes 0 and standard deviation becomes 1
#mathematically we are calculating using the formula z=(x-u)/(p)
#x=original value
#u=colunm mean
#p=colunm standard deviation
#uses : makes sure all features are on the same scale for Distance-based models (KNN, SVM) 
#or Gradient-based models (logistic regression, neural nets)
print("the three colunms : temperature, humidity and precipitation is now standardized")

#save to sql 
df.to_sql("daily_weather_cleaned", engine, schema="manteca_data", if_exists="replace", index=False)
print("Data sent to sql into table 'daily_weather_cleaned'")



'''missing values
    lag/ rolling/ trend - drop or fill defaults
    raw measurements - drop or impute(fill with mean and median)
    bins - drop or fill 'unknown'
date time parsing and indexing - check date time format and index if required
outlier detection - check for thresholds and range limits of things like temperature, humidity
duplicate rows - drop if there are any duplicates
validate data types
feature scaling - normalize or standardize continous features (StandardScaler, fit_transform)
''' 



