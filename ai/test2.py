import pandas as pd
from sqlalchemy import create_engine
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Database connection
engine = create_engine('mysql+pymysql://hokengakari:230073@localhost/shoubou_data')

# Load data from the database
tables = ['【旭】2014.01.01～2016.12.31 (1) 使いそうな奴', 
          '【旭】2017.01.01～2019.12.31使いそうな奴', 
          '【旭】2020.01.01～2020.12.31 使いそうな奴', 
          '【旭】2021.01.01～2022.12.31 使いそうな奴', 
          '【旭】2023.01.01～2024.06.30 使いそうな奴']

# Combine all tables
data_frames = [pd.read_sql_table(table, engine) for table in tables]
data = pd.concat(data_frames)

# Preprocess Date and Time
data['覚知日時'] = pd.to_datetime(data['覚知年月日'] + ' ' + data['覚知時'].astype(str) + ':00')

# Extract year, month, day, hour
data['year'] = data['覚知日時'].dt.year
data['month'] = data['覚知日時'].dt.month
data['day'] = data['覚知日時'].dt.day
data['hour'] = data['覚知日時'].dt.hour

# Encode '天候'
if '天候' in data.columns:
    label_encoder = LabelEncoder()
    data['天候_encoded'] = label_encoder.fit_transform(data['天候'])
else:
    print("Column '天候' is missing. Cannot perform encoding.")

# Encode '出場場所地区'
data['出場場所地区_encoded'] = data['出場場所地区'].astype('category').cat.codes

# Define the target as the count of incidents
data['incident_count'] = 1

# Prepare features and target
X = data[['year', 'month', 'day', 'hour', '天候_encoded', '出場場所地区_encoded']]
y = data['incident_count']

# Initialize TimeSeriesSplit
ts_split = TimeSeriesSplit(n_splits=3)  # Adjust splits based on data size

# Initialize model
model = XGBRegressor()

# Define parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(model, param_grid, cv=ts_split, scoring='neg_mean_absolute_error', n_jobs=-1)

# Fit model
grid_search.fit(X, y)

# Get best model
best_model = grid_search.best_estimator_

# Make predictions
y_pred = best_model.predict(X)

# Calculate MAE, MSE, RMSE
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)

# Print evaluation metrics
print(f"Best Model MAE: {mae}")
print(f"Best Model MSE: {mse}")
print(f"Best Model RMSE: {rmse}")

# Predict future incidents
average_weather = data['天候_encoded'].mean() if '天候_encoded' in data.columns else 0

future_dates = pd.date_range(start='2024-07-01', periods=30, freq='D')

future_data = pd.DataFrame({
    'year': future_dates.year,
    'month': future_dates.month,
    'day': future_dates.day,
    'hour': [0] * 30,
    '天候_encoded': [average_weather] * 30,
    '出場場所地区_encoded': [0] * 30  # Use a placeholder or average value if needed
})

future_predictions = best_model.predict(future_data)

# Store and save future predictions
results = pd.DataFrame({
    'Date': future_dates,
    'Predicted_Incidents': future_predictions
})

results.to_csv('predicted_datas_per_location.csv', index=False)
