import pandas as pd
from sqlalchemy import create_engine
from db_config import get_sqlalchemy_url
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
import seaborn as sns
from prophet import Prophet
from statsmodels.tsa.seasonal import seasonal_decompose #to decompose a time series into its trend, seasonal, and residual components
import numpy as np
from prophet.diagnostics import cross_validation, performance_metrics

engine = create_engine(get_sqlalchemy_url())

query="""
SELECT * FROM manteca_data.daily_weather_cleaned
where date is not null 
order by date asc;    
"""
df=pd.read_sql(query,engine)

#regression model to predict precipitation using lagged and rolling humidity values
#step 1 : Understand Your Data
 #Review column meanings and units
df.info()
#step 2 : Explore Relationships
#Plot precipitation vs. lagged humidity
d = pd.DataFrame(df,columns=["humidity_lag_1","precipitation"])
d.plot(kind='scatter',x="humidity_lag_1", y="precipitation")
plt.show()

#Plot precipitation vs. rolling humidity
d = pd.DataFrame(df,columns=["humidity_rolling_average","precipitation"])
d.plot(kind='scatter',x="humidity_rolling_average", y="precipitation")
plt.show()

#Check correlations between above 2
correlation = df['humidity_lag_1'].corr(df['precipitation'])
print(f"correlation between lagged humidity and precipitation:{correlation:.3f}")
correlation = df['humidity_rolling_average'].corr(df['precipitation'])
print(f"correlation between average humidity and precipitation:{correlation:.3f}")

#step 3 : Prepare the Dataset
#Drop or impute missing values
#done in cleaning program: dropna() 

#Select only the relevant columns
#X=df[["humidity_lag_1"],["humidity_rolling_average"]] wrong
X=df[["humidity_lag_1","humidity_rolling_average"]]
Y=df["precipitation"]
print(X.isnull().sum())
print(Y.isnull().sum())

#Split into training and test sets (e.g., 80/20)
'''X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
general approach. but this being a data in chronological order, we have to do it a bit differently
random_state=42 : This sets the random seed for reproducibility. 
#Without it, you'd get a different split each time you run the code. Using 42 is a common convention, but any integer works.'''
split_index = int(len(df) * 0.8)
X_train = X[:split_index]
X_test = X[split_index:]
y_train = Y[:split_index]
y_test = Y[split_index:]

#Step 4: Build the Model
#Choose linear regression as your starting point
reg=linear_model.LinearRegression()
print("linear regression done")
#Fit the model using training data
reg.fit(X_train, y_train)
print("model fitting done")
#Review the coefficients and intercept
print('coefficients:',reg.coef_)
'''tells how much each feature contributes to the prediction.
For example, if humidity_lag_1 has a coefficient of 0.5, 
then every 1% increase in lagged humidity adds 0.5 units to predicted precipitation.'''
plt.plot(X,reg.predict(X),color='green')
plt.show()

#Step 5: Evaluate the Model
#Predict on test data
prediction = reg.predict(X_test)
print('prediction  : ',prediction)

#Calculate MAE, MSE, and RMSE
'''MAE = Mean absolute error
difference between prediction and actual values'''
mae = mean_absolute_error(y_true = y_test, y_pred=prediction)
print("Mean Absolute Error", mae)
#Lower is better; less sensitive to outliers

#mean squared error : to calcuate the loss
mse = mean_squared_error(y_true = y_test, y_pred=prediction)
print('Mean squared error',mse)
#Penalizes larger errors more heavily

#RMSE : root mean squared error
#says how much data points spread sround the best line
rmse = root_mean_squared_error(y_true = y_test, y_pred=prediction)
print('Root mean squared error',rmse)
#Same units as target variable; intuitive for spread around regression line

#r2 score : measures the proportion of variance in a dependent variable
r2 = r2_score(y_true = y_test, y_pred=prediction)
print("r² Score:", r2)

######################

#Train a simple classifier to label days as rainy or clear.

'''step 1. Define the Classification Target
Create a binary label: 
1 for rainy days (e.g., precipitation > 0)
0 for clear days (e.g., precipitation = 0)
'''
df['is_rainy'] = df['precipitation'].apply(lambda x: 1 if x >= 0.1 else 0)  #wrote 0 initially, but changed to 0.1 for threshold in case What if precipitation == 0.0001?
#Ensure the label is consistent and interpretable.
print(df['precipitation'].isnull().sum())  # Should be 0 ideally

'''step 2. Explore Class Balance
•	Check how many rainy vs. clear days exist.'''
days_count = df['is_rainy'].value_counts()
print(days_count)
#If the classes are imbalanced, consider techniques like resampling or class weighting.
#class weighing
model = LogisticRegression(class_weight='balanced')
#Oversample Rainy Days
rainy = df[df['is_rainy'] == 1]
clear = df[df['is_rainy'] == 0]
rainy_upsampled = resample(rainy, replace=True, n_samples=len(clear), random_state=42)
df_balanced = pd.concat([clear, rainy_upsampled]).reset_index(drop=True)
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
print(df_balanced['is_rainy'].value_counts())

#step 3. Select Predictive Features
#Choose variables that may influence rainfall: 
#Humidity (lagged, rolling) already taken care under humidity_lag_1 and humidity_rolling_average

#scaling Temperature
df['temp_scaled'] = (df['temperature'] - df['temperature'].mean()) / df['temperature'].std()


#Time-based features (season, month, hour)
df['datetime'] = pd.to_datetime(df['date'])  # ensure datetime column exists
df = df.sort_values('datetime').reset_index(drop=True)  # keep column, sorted
#extracting features
df['month'] = df['datetime'].dt.month
df['hour'] = df['datetime'].dt.hour
df['season'] = df['month'] % 12 // 3 + 1 #1-winter, 2-spring etc
df.info()

'''step 4. Preprocess the Data'''
#check missing values in df
sns.heatmap(df.isnull(), cbar=False, yticklabels=False) 
plt.title("Missing Values Heatmap")
plt.show()
#Encode categorical variables (if any). all are in int and float except temperature bin
df_encoded = pd.get_dummies(df, columns=['temperature_bin'], drop_first=True)

#Ensure no data leakage from future values.
# Sort by datetime to preserve temporal order
df_encoded = df_encoded.sort_values(by='datetime')
# Define train-test split index (e.g., 80% train)
split_index = int(len(df_encoded) * 0.8)
# Chronological split
train_df = df_encoded.iloc[:split_index]
test_df = df_encoded.iloc[split_index:]
# Confirm no future leakage
print("Train range:",train_df['datetime'].min(),"to",train_df['datetime'].max())
print("Test range:",test_df['datetime'].min(),"to",test_df['datetime'].max())

 
#step 5. Split the Dataset
Y = df_encoded['is_rainy']  #Binary target
X = df_encoded[[  #Predictors
    'humidity_rolling_average', 'temperature_trend', 'temp_scaled',
    'month', 'hour', 'season'] + [col for col in df_encoded.columns if 'temperature_bin_' in col]]
#Use a chronological split (e.g., 80/20) to preserve time order.
split_index = int(len(df_encoded) * 0.8)
#Separate into training and test sets.
X_train = X[:split_index]
X_test = X[split_index:]
y_train = Y[:split_index]
y_test = Y[split_index:]

print(y_train.unique()) # Should show [0, 1]
print(y_train.dtype)# Should be int64 or int32


 
'''step 6. Choose a Classification Model'''
#Choose linear regression as your starting point
model=LogisticRegression(class_weight='balanced')
model.fit(X_train,y_train)
prediction = model.predict(X_test)
print("Linear regression done")

#decision tree
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train,y_train)
prediction = model.predict(X_test)
print("Decision tree done")

#random forest
model=RandomForestClassifier(n_estimators=100, random_state=42)
#n_estimators : number of individual decision trees that will be built in the forest
model.fit(X_train,y_train)
prediction = model.predict(X_test)
print("random forest done")
 
#Evaluate Performance : Accuracy, Precision, Recall, F1 Score
print('accuracy : ',accuracy_score(y_test, prediction))
print('precision = ',precision_score(y_test,prediction,average='macro'))
print('recall= ',recall_score(y_test,prediction,average='macro'))
print('f1 score = ',f1_score(y_test,prediction,average='macro'))

#Confusion Matrix
cm=confusion_matrix(y_test,prediction)
plt.figure(figsize=(10,4))
sns.heatmap(cm,annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Consider ROC-AUC if probability outputs are available.
y_probs=model.predict_proba(X_test)[:,1] #Get predicted probabilities for the positive class (Rainy = 1)
roc_auc = roc_auc_score(y_test, y_probs)
print(f"ROC-AUC Score: {roc_auc:.2f}")
#plotting curve
fpr,tpr,thresholds=roc_curve(y_test,y_probs)
plt.figure(figsize=(10,4))
plt.plot(fpr,tpr,label = f'AUC= {roc_auc:.2f}')
plt.plot([0,1],[0,1],'k--') #diagonal
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

##########
#Use a time series model (like Prophet or ARIMA) to forecast temperature or precipitation.

'''
 1. **Define the Forecasting Goal**
- Choose your target variable: temperature or precipitation.
- Decide the forecast horizon: next day, week, or month.

'''
#i want to forecast temperature and precipitation for nect week and month
### 5. **Choose the Model**
# - **Prophet**: Best for data with strong seasonality and holidays.
# - **ARIMA**: Best for stationary data (no trend or seasonality unless differenced).
#pip install prophet

#prepare data
temp_df=df[['datetime','temperature']].rename(columns={'datetime':'ds', 'temperature': 'y'})
precip_df=df[['datetime','precipitation']].rename(columns={'datetime':'ds','precipitation':'y'})
#renaming is done to meet the requirement of prophet

#training 
# - Fit the model on historical data.

temp_model=Prophet()
temp_model.fit(temp_df)

precip_model=Prophet()
precip_model.fit(precip_df)

#forecast data
future_temp=temp_model.make_future_dataframe(periods=30) #for 30 days
future_precip=precip_model.make_future_dataframe(periods=30)

#generating forecasts
forecast_temp=temp_model.predict(future_temp)
forecast_precip=precip_model.predict(future_precip)

"""
### 2. **Prepare the Time Series Data**
- Ensure your data is chronologically ordered.
- Use a consistent time interval (e.g., hourly, daily).
- Handle missing timestamps or gaps.
- Convert your date column to proper datetime format.
"""
#data is chronologically ordered (date time)
#daily data already available
#no missing values, already checked
#formatting done already

#Convert to datetime and sort
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)
#set datetime as index
df.set_index('datetime', inplace=True)
#create a complete daily datetime index
full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
#reindex to fill missing days
df = df.reindex(full_index)
df.index.name = 'datetime'
#separate numeric and categorical columns
numeric_cols = df.select_dtypes(include='number').columns
categorical_cols = ['temperature_bin']  
#resample numeric columns (daily mean)
df_numeric = df[numeric_cols].resample('D').mean()
#resample categorical columns (daily mode)
df_categorical = df[categorical_cols].resample('D').agg(
    lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
#combine both
df_daily = pd.concat([df_numeric, df_categorical], axis=1)
#Fill missing values or flag them
df_daily = df_daily.ffill().bfill()  # Or use interpolation if appropriate
df = df.reset_index().set_index('datetime')



"""
### 3. **Visualize the Series**
- Plot the time series to understand trends, seasonality, and noise.
- Identify any anomalies or outliers that may affect modeling.
"""
temp_model.plot(forecast_temp)
plt.title("Temperature forecast")
plt.show()

precip_model.plot(forecast_precip)
plt.title("Precipitation forecast")
plt.show()

# Temperature over time
plt.figure(figsize=(10, 4))
plt.plot(df.index, df['temperature'], label='Temperature')
plt.title("Historical Temperature")
plt.xlabel("Date")
plt.ylabel("Temperature")
plt.grid(True)
plt.legend()
plt.show()

# Precipitation over time
plt.figure(figsize=(10, 4))
plt.plot(df.index, df['precipitation'], label='Precipitation', color='teal')
plt.title("Historical Precipitation")
plt.xlabel("Date")
plt.ylabel("Precipitation")
plt.grid(True)
plt.legend()
plt.show()

# Last 60 days of temperature
df_recent = df[df.index > df.index.max() - pd.Timedelta(days=60)]
plt.figure(figsize=(10, 4))
plt.plot(df_recent.index, df_recent['temperature'], label='Recent Temp')
plt.title("Recent Temperature Trend")
plt.xlabel("Date")
plt.ylabel("Temperature")
plt.grid(True)
plt.legend()
plt.show()

"""

### 4. **Decompose the Series (Optional but Insightful)**
- Break the series into:
  - **Trend**: long-term movement
  - **Seasonality**: repeating patterns (daily, weekly, yearly)
  - **Residuals**: random noise

This helps you choose between Prophet (which handles seasonality well) or ARIMA (which assumes stationarity).
"""
if df.index.name != 'datetime':
    df = df.set_index('datetime')
series = df['temperature'] #choose the column to decompose (temperature)
decomposition = seasonal_decompose(series, model='additive', period=30)  # For monthly seasonality

# Plot the decomposition
plt.figure(figsize=(12, 8))
plt.subplot(411)
plt.plot(decomposition.observed, label='Observed')
plt.legend(loc='upper left')
plt.subplot(412)
plt.plot(decomposition.trend, label='Trend')
plt.legend(loc='upper left')
plt.subplot(413)
plt.plot(decomposition.seasonal, label='Seasonality')
plt.legend(loc='upper left')
plt.subplot(414)
plt.plot(decomposition.resid, label='Residuals')
plt.legend(loc='upper left')
plt.tight_layout()
plt.suptitle("Temperature Series Decomposition", y=1.02)
plt.show()


"""
### 6. **Train the Model**
- Validate using a hold-out set or cross-validation.
- Tune parameters (e.g., ARIMA’s p, d, q or Prophet’s seasonality modes).
"""
df_prophet = df.reset_index()[['datetime', 'temperature']].rename(columns={'datetime': 'ds', 'temperature': 'y'}) #data preparing
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode='additive',
    changepoint_prior_scale=0.5  #tune this to control flexibility
)
model.fit(df_prophet) #model fitting
#cross validation
df_cv = cross_validation(model, initial='300 days', period='30 days', horizon='30 days')
df_perf = performance_metrics(df_cv)
print(df_perf[['horizon', 'mae', 'rmse', 'mape']])
"""
### 7. **Generate Forecasts**
- Predict future values for your chosen horizon.
- Include confidence intervals to express uncertainty."""
#next 30 dats
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)
#plotting
model.plot(forecast)
plt.title("Temperature Forecast with Prophet")
plt.show()
#plotting components
model.plot_components(forecast)
plt.suptitle("Seasonality and Trend Components", y=1.02)
plt.show()
"""
### 8. **Evaluate Forecast Accuracy**
- Use metrics like:
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - MAPE (Mean Absolute Percentage Error)

Compare actual vs. predicted values visually and numerically.
"""
#compare prediction and actual for 30 days
actual = df_prophet['y'].iloc[-30:].values
predicted = forecast['yhat'].iloc[-30:].values

mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))
mape = np.mean(np.abs((actual - predicted) / actual)) * 100
print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
"""
### 9. **Refine and Iterate**
- Try different seasonalities (weekly, yearly).
- Add external regressors (e.g., humidity, wind speed).
- Re-train periodically as new data arrives."""
#another seasonality model
model_refined = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    seasonality_mode='multiplicative',
    changepoint_prior_scale=0.2
)
#adding external regressors
df_prophet['humidity'] = df.reset_index()['humidity']
model_refined.add_regressor('humidity')
#fitting and forecasting
model_refined.fit(df_prophet)
future = model_refined.make_future_dataframe(periods=30)
future['humidity'] = pd.concat([
    df['humidity'],
    pd.Series([df['humidity'].iloc[-1]] * 30)
], ignore_index=True)
forecast_refined = model_refined.predict(future)
"""### 10. **Deploy or Share Results**
- Export forecasts to dashboards or reports.
- Use them to inform decisions (e.g., weather alerts, resource planning).
"""

# Save forecast to SQL with readable headers
forecast = forecast.rename(columns={
    "ds": "date",
    "yhat": "predicted_temperature",
    "yhat_lower": "predicted_temp_min",
    "yhat_upper": "predicted_temp_max"
})
forecast[["date", "predicted_temperature", "predicted_temp_min", "predicted_temp_max"]].to_sql(
    "temperature_forecast",
    engine,
    schema="manteca_data",
    if_exists="replace",
    index=False
)

print("Data sent to SQL into table 'temperature_forecast'")


# Example: Get predictions for a specific month (e.g., May 2024)
future_dates = pd.date_range(start="2024-05-01", end="2024-05-31", freq="D")
future_df = pd.DataFrame({"ds": future_dates})
future_df["humidity"] = df["humidity"].iloc[-1]  # placeholder for regressors

forecast_specific = model.predict(future_df)
forecast_specific = forecast_specific.rename(columns={
    "ds": "date",
    "yhat": "predicted_temperature",
    "yhat_lower": "predicted_temp_min",
    "yhat_upper": "predicted_temp_max"
})
print(forecast_specific[["date", "predicted_temperature", "predicted_temp_min", "predicted_temp_max"]])


#Visual summary
plt.figure(figsize=(10, 4))
plt.plot(df_prophet['ds'], df_prophet['y'], label='Actual')
plt.plot(forecast['date'], forecast['predicted_temperature'], label='Forecast')
plt.fill_between(forecast['date'], forecast['predicted_temp_min'], forecast['predicted_temp_max'], alpha=0.2)
plt.title("Temperature Forecast with Confidence Intervals")
plt.xlabel("Date")
plt.ylabel("Temperature")
plt.legend()
plt.grid(True)
plt.show()