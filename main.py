import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from prepare_yerevan_data import prepare_yerevan_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error


# In this file we're training a model to predict the pm2.5 level only
air_data = prepare_yerevan_data()
air_data = air_data.drop(columns=['pm10', 'nitrogen_dioxide'])

pd.set_option("display.max_columns", 20)

corr_matrix = air_data.corr()
print(corr_matrix["pm2_5_delta"].sort_values(ascending=False))

# Spltting into training and testing
# TODO: Clean up this part a little bit
X = air_data.drop(columns=['pm2_5_delta', 'time', 'pm2_5'])
y = air_data['pm2_5_delta']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42, shuffle=False)


# Selecting a model and fitting
# TODO: Perform a grid search to tune the hyperparams
# TODO: Scale the data maybe?
# model = RandomForestRegressor(random_state=42)
# model.fit(air_data_train_X, air_data_train_Y)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Predicting
y_pred = model.predict(X_test)

# Evaluating
print(f"MAE: {mean_absolute_error(y_test, y_pred):.3f}")
print(f"RMSE: {root_mean_squared_error(y_test, y_pred):.3f}")
print(f"R^2: {r2_score(y_test, y_pred):.3f}")

# Feature importances for our model
importances = model.feature_importances_
feature_names = X_train.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
print(feature_importance_df.sort_values(by='Importance', ascending=False))
