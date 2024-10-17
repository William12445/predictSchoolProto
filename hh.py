import pandas as pd
import random
from datetime import datetime, timedelta

# Original rainfall data
data = {
    "region": [
        "さちが丘", "万騎が原", "三反田町", "今宿", "今宿南町", "今宿東町",
        "今宿町", "今宿西町", "今川町", "上川井町", "上白根", "上白根町",
        "下川井町", "中尾", "中希望が丘", "中沢", "中白根", "二俣川",
        "南希望が丘", "南本宿町", "善部町", "四季美台", "大池町", "小高町",
        "川井宿町", "川井本町", "川島町", "川島町2", "左近山", "市沢町",
        "本宿町", "本村町", "東希望が丘", "柏町", "桐が作", "白根",
        "白根町", "矢指町", "笹野台", "若葉台", "西川島町", "都岡町",
        "金が谷", "鶴ヶ峰", "鶴ヶ峰本町"
    ],
    "rainfall": [
        30, 70, 20, 100, 60, 40, 50, 80, 90, 10,
        30, 20, 50, 60, 70, 80, 90, 100, 30, 50,
        20, 10, 80, 40, 70, 90, 60, 30, 50, 40,
        80, 90, 10, 20, 50, 30, 70, 60, 80, 40,
        90, 50, 20, 70, 100
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Set the starting date
start_date = datetime(2024, 10, 1)

# Prepare a list to hold the new data
new_data = []

# Generate data for 3 days
for day in range(3):
    date = start_date + timedelta(days=day)
    for index, row in df.iterrows():
        # Randomize rainfall value (for demonstration, you can use the original)
        randomized_rainfall = random.choice(df['rainfall'].values)
        new_data.append({"date": date.strftime("%Y-%m-%d"), "region": row['region'], "rainfall": randomized_rainfall})

# Create a new DataFrame from the new data
new_df = pd.DataFrame(new_data)

# Save to CSV
new_df.to_csv('date.csv', index=False)

print("New CSV file 'rainfall_data_with_dates.csv' created with random data for 3 days.")
