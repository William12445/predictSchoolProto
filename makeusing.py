import pandas as pd
from sqlalchemy import create_engine

# Database connection string
db_connection_str = 'mysql+pymysql://hokengakari:230073@localhost/shoubou_data'

# Create a database connection
engine = create_engine(db_connection_str)

# Define the SQL query
query = """
SELECT 覚知年月日, 覚知曜日, 天候, 出場場所地区, 性別, 年齢区分_サーベイランス用 
FROM all_datas
"""

# Read data into a DataFrame
df = pd.read_sql(query, engine)

# Save the DataFrame to a CSV file
df.to_csv('using_datas.csv', index=False, encoding='utf-8-sig')

print("Data saved to using_datas.csv")
