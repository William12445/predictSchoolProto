import pandas as pd
import pymysql
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

# Database connection details
db_connection_str = 'mysql+pymysql://hokengakari:230073@localhost/shoubou_data'
engine = create_engine(db_connection_str)

# List of tables
tables = [
    '【旭】2014.01.01～2016.12.31 (1) 使いそうな奴',
    '【旭】2017.01.01～2019.12.31使いそうな奴',
    '【旭】2020.01.01～2020.12.31 使いそうな奴',
    '【旭】2021.01.01～2022.12.31 使いそうな奴',
    '【旭】2023.01.01～2024.06.30 使いそうな奴'
]

# Load and combine data from all tables
data_frames = []
for table in tables:
    query = f"""
        SELECT `覚知年月日`, `覚知時`, `出場場所地区` 
        FROM `{table}`
    """
    df = pd.read_sql(query, engine)
    data_frames.append(df)

# Combine all data into a single DataFrame
data = pd.concat(data_frames, ignore_index=True)

# Preprocessing
data['date'] = pd.to_datetime(data['覚知年月日'])  # Convert to datetime
data['day_of_week'] = data['date'].dt.dayofweek  # Extract day of the week
data['year'] = data['date'].dt.year  # Extract year
data['month'] = data['date'].dt.month  # Extract month
data['day'] = data['date'].dt.day  # Extract day of the month
data['hour'] = pd.to_numeric(data['覚知時'], errors='coerce')  # Ensure hour is numeric
data = data.dropna(subset=['hour'])  # Remove rows with invalid hours
data['location'] = data['出場場所地区']  # Location

# Aggregating data to count number of accidents per hour of the day
hourly_counts = data.groupby(['year', 'month', 'day', 'day_of_week', 'hour', 'location']).size().reset_index(name='accident_count')

# Encoding location (assuming categorical data)
hourly_counts = pd.get_dummies(hourly_counts, columns=['location'])

# Features and target
feature_cols = ['year', 'month', 'day', 'day_of_week', 'hour'] + [col for col in hourly_counts.columns if 'location_' in col]
X = hourly_counts[feature_cols]
y = hourly_counts['accident_count']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)

# Model training with MLPRegressor
model = MLPRegressor(hidden_layer_sizes=(300, 100), max_iter=1000, random_state=50)
model.fit(X_train, y_train)

# Prediction
predicted_counts = model.predict(X_test)
mae = mean_absolute_error(y_test, predicted_counts)
print(f'Mean Absolute Error: {mae}')

# Training loss at each iteration
plt.plot(model.loss_curve_)
plt.title('Training Loss Curve')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.show()

# Total count in each location
location_totals = data.groupby('location').size().reset_index(name='total_count')
print("Total count in each location:")
print(location_totals)

# Timetable generation for future days
future_dates = pd.date_range(start=datetime.date.today(), periods=7)
predicted_timetable = []

# Prepare feature columns for future dates
for date in future_dates:
    year = date.year
    month = date.month
    day = date.day
    day_of_week = date.weekday()
    for hour in range(24):
        # Create a DataFrame for each location
        for location in tqdm([col for col in hourly_counts.columns if 'location_' in col], desc='Processing Locations'):
            # Create a DataFrame for prediction with the same columns as X_train
            future_features = pd.DataFrame([[year, month, day, day_of_week, hour] + [1 if col == location else 0 for col in feature_cols if 'location_' in col]],
                                           columns=feature_cols)
            # Predict using the model
            predictions = model.predict(future_features)
            predicted_timetable.append((date, hour, location, predictions[0]))

# Convert to DataFrame for better visualization
timetable_df = pd.DataFrame(predicted_timetable, columns=['Date', 'Hour', 'Location', 'Predicted Accident Count'])

# Save the timetable to a CSV file
timetable_df.to_csv('predicted_accidents_timetable.csv', index=False)

print(f"Timetable saved to 'predicted_accidents_timetable.csv'.")
