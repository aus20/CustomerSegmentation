import os
import pandas as pd

# make true pathway names for data/raw
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_raw_path = os.path.join(project_dir, 'data/raw')
os.makedirs(data_raw_path, exist_ok=True)

# construct the full path to the CSV file
csv_file_path = os.path.join(data_raw_path, 'marketing_campaign.csv')

# read data
df = pd.read_csv(csv_file_path)

# display the features of the dataset
print(df.head())

