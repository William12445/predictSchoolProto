import pandas as pd
from sqlalchemy import create_engine

# Database connection string
db_connection_str = 'mysql+pymysql://hokengakari:230073@localhost/shoubou_data'
engine = create_engine(db_connection_str)

# List of tables to read
tables = [
    '【旭】2014.01.01～2016.12.31 (1) 使いそうな奴',
    '【旭】2017.01.01～2019.12.31使いそうな奴',
    '【旭】2020.01.01～2020.12.31 使いそうな奴',
    '【旭】2021.01.01～2022.12.31 使いそうな奴',
    '【旭】2023.01.01～2024.06.30 使いそうな奴'
]

# Initialize an empty list to hold DataFrames
dfs = []

# Loop through the tables and read each one into a DataFrame
for table in tables:
    print(f"Reading table: {table}")
    df = pd.read_sql_table(table, engine)
    dfs.append(df)

# Concatenate all DataFrames into a single DataFrame
combined_df = pd.concat(dfs, ignore_index=True)

# Export the combined DataFrame to a CSV file
combined_df.to_csv('combined_data.csv', index=False)

print("Data has been successfully exported to 'combined_data.csv'")
