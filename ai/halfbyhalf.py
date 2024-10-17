import pandas as pd
import pymysql
from sqlalchemy import create_engine
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Database connection details
db_connection_str = 'mysql+pymysql://hokengakari:230073@localhost/shoubou_data'
engine = create_engine(db_connection_str)

# List of tables for 2014 to 2020 data
training_tables = [
    '【旭】2014.01.01～2016.12.31 (1) 使いそうな奴',
    '【旭】2017.01.01～2019.12.31使いそうな奴',
    '【旭】2020.01.01～2020.12.31 使いそうな奴'
]

# Load and combine training data from all tables
training_data_frames = []
for table in training_tables:
    query = f"""
        SELECT `覚知年月日`, `覚知時`, `出場場所地区` 
        FROM `{table}`
    """
    df = pd.read_sql(query, engine)
    training_data_frames.append(df)

# Combine all training data into a single DataFrame
training_data = pd.concat(training_data_frames, ignore_index=True)

# Preprocessing training data
training_data['date'] = pd.to_datetime(training_data['覚知年月日'])  # Convert to datetime
training_data['day_of_week'] = training_data['date'].dt.dayofweek  # Extract day of the week
training_data['year'] = training_data['date'].dt.year  # Extract year
training_data['month'] = training_data['date'].dt.month  # Extract month
training_data['day'] = training_data['date'].dt.day  # Extract day of the month
training_data['hour'] = pd.to_numeric(training_data['覚知時'], errors='coerce')  # Ensure hour is numeric
training_data = training_data.dropna(subset=['hour'])  # Remove rows with invalid hours
training_data['location'] = training_data['出場場所地区']  # Location

# Aggregating data to count number of accidents per hour of the day
training_counts = training_data.groupby(['year', 'month', 'day', 'day_of_week', 'hour', 'location']).size().reset_index(name='accident_count')

# Encoding location (assuming categorical data)
training_counts = pd.get_dummies(training_counts, columns=['location'])

# Features and target for training
feature_cols = ['year', 'month', 'day', 'day_of_week', 'hour'] + [col for col in training_counts.columns if 'location_' in col]
X_train = training_counts[feature_cols]
y_train = training_counts['accident_count']

# Model training with MLPRegressor using cross-validation for predictions
model = MLPRegressor(hidden_layer_sizes=(300, 100), max_iter=1000, random_state=50)

# KFold cross-validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Using cross-validation to get predictions for training data
predicted_train_counts = cross_val_predict(model, X_train, y_train, cv=kf)

# Calculate cross-validation MAE on training data
cv_mae = mean_absolute_error(y_train, predicted_train_counts)
print(f'Mean Absolute Error from cross-validation (正解度): {cv_mae}')

# After cross-validation, fit the final model on the entire training dataset
model.fit(X_train, y_train)

# Load 2021-2022 test data
test_table = '【旭】2021.01.01～2022.12.31 使いそうな奴'
test_query = f"""
    SELECT `覚知年月日`, `覚知時`, `出場場所地区` 
    FROM `{test_table}`
"""
test_data = pd.read_sql(test_query, engine)

# Preprocessing test data
test_data['date'] = pd.to_datetime(test_data['覚知年月日'])  # Convert to datetime
test_data['day_of_week'] = test_data['date'].dt.dayofweek  # Extract day of the week
test_data['year'] = test_data['date'].dt.year  # Extract year
test_data['month'] = test_data['date'].dt.month  # Extract month
test_data['day'] = test_data['date'].dt.day  # Extract day of the month
test_data['hour'] = pd.to_numeric(test_data['覚知時'], errors='coerce')  # Ensure hour is numeric
test_data = test_data.dropna(subset=['hour'])  # Remove rows with invalid hours
test_data['location'] = test_data['出場場所地区']  # Location

# Aggregating test data to count number of accidents per hour of the day
test_counts = test_data.groupby(['year', 'month', 'day', 'day_of_week', 'hour', 'location']).size().reset_index(name='accident_count')

# Encoding location for test data (same as training)
test_counts = pd.get_dummies(test_counts, columns=['location'])

# Align test columns with training columns (handling any missing columns)
for col in feature_cols:
    if col not in test_counts.columns:
        test_counts[col] = 0

X_test = test_counts[feature_cols]
y_test = test_counts['accident_count']

# Prediction for 2021-2022 data using the trained model
predicted_counts = model.predict(X_test)

# Calculate accuracy (正解度) on test data
test_mae = mean_absolute_error(y_test, predicted_counts)
print(f'Mean Absolute Error on test data (正解度): {test_mae}')

# Create a DataFrame to compare actual vs predicted values
results_df = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': predicted_counts
})

# Save the results to a CSV file
results_df.to_csv('forecast_vs_actuals.csv', index=False)

# Plotting the actual vs predicted values
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Actual', alpha=0.7)
plt.plot(predicted_counts, label='Predicted', alpha=0.7)
plt.legend()
plt.title('Actual vs Predicted Accident Counts (2021-2022)')
plt.xlabel('Samples')
plt.ylabel('Accident Counts')
plt.show()
