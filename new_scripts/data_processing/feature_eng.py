import pandas as pd
import numpy as np
import os

# load the cleaned dataset
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
data_cleaned_path = os.path.join(project_dir, 'data/new_processed/cleaned_marketing_campaign.csv')
df_cleaned = pd.read_csv(data_cleaned_path)

# summing up the expenditures across different product categories to get the total spend for each customer
df_cleaned['TotalSpend'] = df_cleaned[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].sum(axis=1)

# summing up the number of purchases made through different channels to get the total purchases for each customer
df_cleaned['TotalPurchases'] = df_cleaned[['NumDealsPurchases', 'NumWebPurchases', 'NumStorePurchases', 'NumCatalogPurchases']].sum(axis=1)

df_cleaned['AvgPurchaseValue'] = df_cleaned['TotalSpend'] / df_cleaned['TotalPurchases']

# converting days into years for better interpretation
df_cleaned['RecencyYears'] = df_cleaned['Recency'] / 365

# ensure Dt_Customer is in datetime format
df_cleaned['Dt_Customer'] = pd.to_datetime(df_cleaned['Dt_Customer'])
# reference date covid 19
reference_date = pd.to_datetime('2019-11-17')
# how many years has the customer been with the company
df_cleaned['CustomerSinceYears'] = (reference_date - df_cleaned['Dt_Customer']).dt.days / 365
# how many years has the customer been active
df_cleaned['CustomerSinceYears'] = df_cleaned['CustomerSinceYears'] - df_cleaned['RecencyYears']
# average customer lifespan
avg_customer_lifespan = df_cleaned['CustomerSinceYears'].mean()

# how often does the customer make a purchase
df_cleaned['AvgFrequencyRate'] = df_cleaned['TotalPurchases'] / df_cleaned['CustomerSinceYears']

# customer value is calculated by multiplying the average purchase value by the average frequency rate
df_cleaned['CustomerValue'] = df_cleaned['AvgPurchaseValue'] * df_cleaned['AvgFrequencyRate']

# CLV measures the total revenue a company can expect from a customer throughout their entire relationship
df_cleaned['CLV'] = df_cleaned['CustomerValue'] * avg_customer_lifespan

# Income Bracket
df_cleaned['IncomeBracket'] = pd.cut(df_cleaned['Income'], bins=[0, 0.25, 0.5, 0.75, 1], labels=['Low', 'Middle', 'High', 'Very High'])

# Age Group
df_cleaned['AgeGroup'] = pd.cut(2019 - df_cleaned['Year_Birth'], bins=[0, 30, 50, 70, np.inf], labels=['Young', 'Middle-Aged', 'Senior', 'Elder'])

# Step 5: Save the Updated Dataset
data_final_path = os.path.join(project_dir, 'data/final/feature_engineered_data.csv')
os.makedirs(os.path.dirname(data_final_path), exist_ok=True)
df_cleaned.to_csv(data_final_path, index=False)

print("Feature engineering completed and data saved.")
