import pandas as pd
import numpy as np

def prepare_yerevan_data():

    features = pd.read_csv('yerevan_weather_features.csv')
    targets = pd.read_csv('yerevan_air_quality_targets.csv')

    air_dataset = pd.merge(features, targets, on='time') # Combine the two datasets on the column 'time'

    # Adding new features
    air_dataset['time'] = pd.to_datetime(air_dataset['time'])  # Convert from string to datetime type
    air_dataset['hour'] = air_dataset['time'].dt.hour
    air_dataset['is_weekday'] = air_dataset['time'].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)
    air_dataset['is_busy'] = air_dataset['hour'].apply(lambda y: 1 if (y >= 7 and y <= 10) or
                                                                      (y >= 17 and y <= 20) else 0)

    air_dataset['pm2_5_lag_1h'] = air_dataset['pm2_5'].shift(1) # Add a column containing the pm-value in the previous hour
    air_dataset['pm2_5_delta'] = air_dataset['pm2_5'] - air_dataset['pm2_5_lag_1h']

    air_dataset['hour_sin'] = np.sin(2 * np.pi * air_dataset['hour'] / 24)
    air_dataset['hour_cos'] = np.cos(2 * np.pi * air_dataset['hour'] / 24)
    air_dataset['stagnation'] = air_dataset['relative_humidity_2m'] / (air_dataset['wind_speed_10m'] + 1)

    air_dataset['is_raining'] = air_dataset['precipitation'].apply(lambda z: 1 if z > 0 else 0)

    air_dataset['stagnation'] = np.log1p(air_dataset['stagnation'])
    air_dataset['wind_speed_10m'] = np.log1p(air_dataset['wind_speed_10m'])

    air_dataset = air_dataset.dropna() # The very first row doesn't have a previous, so we drop it

    return air_dataset



