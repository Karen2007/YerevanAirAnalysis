import pandas as pd
import numpy as np

def prepare_yerevan_data_pm_2_5():

    features = pd.read_csv('yerevan_weather_features.csv')
    targets = pd.read_csv('yerevan_air_quality_targets.csv')

    air_data = pd.merge(features, targets, on='time') # Combine the two datasets on the column 'time'

    # Adding new features
    air_data['time'] = pd.to_datetime(air_data['time'])  # Convert from string to datetime type
    air_data['hour'] = air_data['time'].dt.hour
    # air_data['is_weekday'] = air_data['time'].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)
    air_data['is_busy'] = air_data['hour'].apply(lambda y: 1 if (y >= 7 and y <= 10) or
                                                                      (y >= 17 and y <= 20) else 0)

    # Current PM2.5 level is highly dependent on what it was an hour,
    # couple of hours ago, or even at the same time yesterday
    air_data['pm2_5_lag_1h'] = air_data['pm2_5'].shift(1)
    air_data['pm2_5_lag_2h'] = air_data['pm2_5'].shift(2)
    air_data['pm2_5_lag_3h'] = air_data['pm2_5'].shift(3)
    air_data['pm2_5_lag_1d'] = air_data['pm2_5'].shift(24)
    air_data['pm2_5_delta'] = air_data['pm2_5'] - air_data['pm2_5_lag_1h']

    # Transform the hour column into a sinusoidal value,
    # to make for example 23 and 1 more related to each other
    air_data['hour_sin'] = np.sin(2 * np.pi * air_data['hour'] / 24)
    air_data['hour_cos'] = np.cos(2 * np.pi * air_data['hour'] / 24)
    air_data = air_data.drop(columns=['hour'])

    air_data['stagnation'] = air_data['relative_humidity_2m'] / (air_data['wind_speed_10m'] + 1)

    # Apply log transform to make them look like bell shaped curves
    air_data['stagnation'] = np.log1p(air_data['stagnation'])
    air_data['wind_speed_10m'] = np.log1p(air_data['wind_speed_10m'])

    # Wind takes time to play an effect so we add lag variables
    air_data['wind_speed_lag_1h'] = air_data['wind_speed_10m'].shift(1)
    air_data['wind_speed_lag_2h'] = air_data['wind_speed_10m'].shift(2)
    air_data['wind_speed_lag_3h'] = air_data['wind_speed_10m'].shift(3)

    air_data['temperature_rolling_3h_mean'] = air_data['temperature_2m'].shift(1).rolling(3).mean()
    air_data['temperature_rolling_6h_mean'] = air_data['temperature_2m'].shift(1).rolling(6).mean()
    air_data['temperature_rolling_12h_mean'] = air_data['temperature_2m'].shift(1).rolling(12).mean()
    air_data['temperature_rolling_24h_mean'] = air_data['temperature_2m'].shift(1).rolling(24).mean()

    # Add some rolling window variables
    air_data['pm2_5_rolling_6h_mean'] = air_data['pm2_5'].shift(1).rolling(6).mean()
    air_data['pm2_5_rolling_6h_std'] = air_data['pm2_5'].shift(1).rolling(6).std()
    air_data['pm2_5_rolling_12h_mean'] = air_data['pm2_5'].shift(1).rolling(12).mean()

    # Trying out some new variables
    air_data['temp_humidity_2m'] = air_data['temperature_2m'] * air_data['relative_humidity_2m']
    air_data['temperature_pressure'] = air_data['temperature_2m'] * air_data['surface_pressure']

    air_data['temp_humidity_rolling_3h'] = air_data['temp_humidity_2m'].shift(1).rolling(3).mean()

    air_data = air_data.dropna() # The first few rows don't have previous data, so we drop them

    air_data = air_data.drop(columns=['pm10', 'nitrogen_dioxide',  'time', 'pm2_5'])

    return air_data

def prepare_yerevan_data_pm_10():

    features = pd.read_csv('yerevan_weather_features.csv')
    targets = pd.read_csv('yerevan_air_quality_targets.csv')

    air_data = pd.merge(features, targets, on='time') # Combine the two datasets on the column 'time'

    # Adding new features
    air_data['time'] = pd.to_datetime(air_data['time'])  # Convert from string to datetime type
    air_data['hour'] = air_data['time'].dt.hour
    air_data['is_busy'] = air_data['hour'].apply(lambda y: 1 if (y >= 7 and y <= 10) or
                                                                      (y >= 17 and y <= 20) else 0)

    # Current PM2.5 level is highly dependent on what it was an hour,
    # couple of hours ago, or even at the same time yesterday
    air_data['pm10_lag_1h'] = air_data['pm10'].shift(1)
    air_data['pm10_lag_2h'] = air_data['pm10'].shift(2)
    air_data['pm10_lag_3h'] = air_data['pm10'].shift(3)
    air_data['pm10_lag_1d'] = air_data['pm10'].shift(24)
    air_data['pm10_delta'] =  air_data['pm10'] - air_data['pm10_lag_1h']

    # Transform the hour column into a sinusoidal value,
    # to make for example 23 and 1 more related to each other
    air_data['hour_sin'] = np.sin(2 * np.pi * air_data['hour'] / 24)
    air_data['hour_cos'] = np.cos(2 * np.pi * air_data['hour'] / 24)
    air_data = air_data.drop(columns=['hour'])

    air_data['stagnation'] = air_data['relative_humidity_2m'] / (air_data['wind_speed_10m'] + 1)

    # Apply log transform to make them look like bell shaped curves
    air_data['stagnation'] = np.log1p(air_data['stagnation'])
    air_data['wind_speed_10m'] = np.log1p(air_data['wind_speed_10m'])

    # Wind takes time to play an effect so we add lag variables
    air_data['wind_speed_lag_1h'] = air_data['wind_speed_10m'].shift(1)
    air_data['wind_speed_lag_2h'] = air_data['wind_speed_10m'].shift(2)
    air_data['wind_speed_lag_3h'] = air_data['wind_speed_10m'].shift(3)

    air_data['wind_speed_delta'] = air_data['wind_speed_10m'] - air_data['wind_speed_lag_1h']

    # Add some rolling window variables
    air_data['temperature_rolling_3h_mean'] = air_data['temperature_2m'].shift(1).rolling(3).mean()
    air_data['temperature_rolling_6h_mean'] = air_data['temperature_2m'].shift(1).rolling(6).mean()
    air_data['temperature_rolling_12h_mean'] = air_data['temperature_2m'].shift(1).rolling(12).mean()
    air_data['temperature_rolling_24h_mean'] = air_data['temperature_2m'].shift(1).rolling(24).mean()

    air_data['pm10_rolling_6h_mean'] = air_data['pm10'].shift(1).rolling(6).mean()
    air_data['pm10_rolling_6h_std'] = air_data['pm10'].shift(1).rolling(6).std()
    air_data['pm10_rolling_12h_mean'] = air_data['pm10'].shift(1).rolling(12).mean()

    air_data['wind_volatility_interaction'] = air_data['wind_speed_10m'] * air_data['pm10_rolling_6h_std']

    # Trying out some new variables
    air_data['temp_humidity_2m'] = air_data['temperature_2m'] * air_data['relative_humidity_2m']
    air_data['temperature_pressure'] = air_data['temperature_2m'] * air_data['surface_pressure']

    air_data['temp_humidity_rolling_3h'] = air_data['temp_humidity_2m'].shift(1).rolling(3).mean()

    air_data = air_data.dropna() # The first few rows don't have previous data, so we drop them

    air_data = air_data.drop(columns=['pm10', 'nitrogen_dioxide',  'time', 'pm2_5'])

    return air_data

def prepare_yerevan_data_nitrogen_dioxide():
    features = pd.read_csv('yerevan_weather_features.csv')
    targets = pd.read_csv('yerevan_air_quality_targets.csv')

    air_data = pd.merge(features, targets, on='time') # Combine the two datasets on the column 'time'

    air_data['wind_speed_10m'] = np.log1p(air_data['wind_speed_10m'])
    air_data['pm2_5'] = np.log1p(air_data['pm2_5'])
    air_data['pm10'] = np.log1p(air_data['pm10'])

    air_data = air_data.drop(columns=['time'])

    return air_data