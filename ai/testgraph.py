import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load the data
data = pd.read_csv('combined_data.csv')

# Print data summary
print(data.head())
print(data.columns)

# Convert date column to datetime
data['覚知年月日'] = pd.to_datetime(data['覚知年月日'])

# Extract year, month, day, and day of the week
data['Year'] = data['覚知年月日'].dt.year
data['Month'] = data['覚知年月日'].dt.month
data['Day'] = data['覚知年月日'].dt.day
data['Day_of_Week'] = data['覚知年月日'].dt.day_name()  # Extract day names

# Print to confirm columns
print(data[['Year', 'Month', 'Day', 'Day_of_Week']].head())

# Aggregate data to count incidents per location per day
aggregated_data = data.groupby(['出場場所地区', 'Year', 'Month', 'Day']).size().reset_index(name='Incident_Count')

# Print aggregated data to check
print(aggregated_data.head())

# Encode categorical data
label_encoders = {}
for column in ['出場場所地区', 'Day_of_Week']:
    if column in aggregated_data.columns:
        le = LabelEncoder()
        aggregated_data[column] = le.fit_transform(aggregated_data[column])
        label_encoders[column] = le
    else:
        print(f'Column {column} not found in aggregated_data')

# Set the target column name
target_column = 'Incident_Count'

# Split features and target
X = aggregated_data.drop(target_column, axis=1)
y = aggregated_data[target_column]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose a model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate MAE
mae = mean_absolute_error(y_test, y_pred)
print(f'Overall MAE: {mae}')

# Example prediction for future dates
future_dates = pd.date_range(start='2024-01-01', end='2024-12-31')
future_data = pd.DataFrame({
    'Year': future_dates.year,
    'Month': future_dates.month,
    'Day': future_dates.day,
    'Day_of_Week': future_dates.day_name()  # Get day names as strings
})

# Encode categorical data in future data
for column in ['Day_of_Week']:
    if column in label_encoders:
        if not future_data[column].isin(label_encoders[column].classes_).all():
            raise ValueError(f"Future data contains unseen labels in {column}")
        future_data[column] = label_encoders[column].transform(future_data[column])
    else:
        print(f'Column {column} not found in label_encoders')

# Ensure all locations are included in future data
locations = aggregated_data['出場場所地区'].unique()
future_predictions = []

for location in locations:
    future_location_data = future_data.copy()
    future_location_data['出場場所地区'] = location
    
    # Encode location data
    if location not in label_encoders['出場場所地区'].classes_:
        raise ValueError(f"Location {location} not found in label_encoders['出場場所地区']")
        
    future_location_data['出場場所地区'] = label_encoders['出場場所地区'].transform([location] * len(future_location_data))
    
    # Predict incident counts for the location
    X_future = future_location_data[X.columns]
    predictions = model.predict(X_future)
    
    # Add predictions to future_data
    future_location_data['Predicted_Incident_Count'] = predictions
    future_location_data['Location'] = location
    future_predictions.append(future_location_data)

# Combine all future predictions
all_predictions = pd.concat(future_predictions)

# Aggregate predictions by location and date
location_predictions = all_predictions.groupby(['Location', 'Year', 'Month', 'Day']).agg({
    'Predicted_Incident_Count': 'sum'
}).reset_index()

# Print predictions
print(location_predictions.head())

# Save predictions to CSV for further analysis
location_predictions.to_csv('predicted_incidents_by_location.csv', index=False)
ijui5r1