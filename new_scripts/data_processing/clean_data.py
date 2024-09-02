import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import numpy as np

# make true pathway names for data/raw
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
data_raw_path = os.path.join(project_dir, 'data/raw')
os.makedirs(data_raw_path, exist_ok=True)

# construct the full path to the CSV file
csv_file_path = os.path.join(data_raw_path, 'marketing_campaign.csv')

df = pd.read_csv(csv_file_path, delimiter= '\t')

df = df.dropna()  # drop rows with missing values

# drop unnecessary columns (in my case)
columns_to_drop = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5','Z_CostContact', 'Z_Revenue']
df_cleaned = df.drop(columns=columns_to_drop)

# increasing all birth years by 10 
df_cleaned['Year_Birth'] = df_cleaned['Year_Birth'] + 10

# convert Dt_Customer to datetime format to do time-based operations
df_cleaned['Dt_Customer'] = pd.to_datetime(df_cleaned['Dt_Customer'], dayfirst=True)

# verify the conversion
print(df_cleaned['Dt_Customer'].head())

# outlier detection and removal using z-score
# threshold = 3 because we want to remove the values that are highly unusual (captures 99.7% of the data)
# z-score = how many standard deviations a data point is from the mean
df_cleaned = df_cleaned[(np.abs(stats.zscore(df_cleaned['Income'])) < 3)]

# save original income values for future reference
df_cleaned['IncomeRaw'] = df_cleaned['Income']

# normalize the income values
scaler = MinMaxScaler()
df_cleaned[['Income']] = \
    scaler.fit_transform(df_cleaned[['IncomeRaw']])

# simplify the marital status
df_cleaned["Living_With"] = df_cleaned["Marital_Status"].replace({
    "Married": "Partner", 
    "Together": "Partner", 
    "Absurd": "Alone", 
    "Widow": "Alone", 
    "YOLO": "Alone", 
    "Divorced": "Alone", 
    "Single": "Alone"
})

# total children living in the household
df_cleaned["Children"] = df_cleaned["Kidhome"] + df_cleaned["Teenhome"]

# total members in the household
df_cleaned["Family_Size"] = df_cleaned["Living_With"].replace({"Alone": 1, "Partner": 2}) + df_cleaned["Children"]

# indicating parenthood
df_cleaned["Is_Parent"] = np.where(df_cleaned["Children"] > 0, 1, 0)

# simplify the education status (not sure if this is needed)
df_cleaned["Education"] = df_cleaned["Education"].replace({
    "Basic": "Undergraduate",
    "2n Cycle": "Undergraduate", 
    "Graduation": "Graduate", 
    "Master": "Postgraduate", 
    "PhD": "Postgraduate"
})

# dropping redundant features
columns_to_drop = ["Marital_Status", "Kidhome", "Teenhome"]
df_cleaned = df_cleaned.drop(columns=columns_to_drop, axis=1)

# verify the normalization
print(df_cleaned.head())
df_cleaned.describe()

data_new_processed_path = os.path.join(project_dir, 'data/new_processed/cleaned_marketing_campaign.csv')

# save the cleaned dataset
df_cleaned.to_csv(data_new_processed_path, index=False)

print("Data cleaning and some feature engineering completed and saved.")