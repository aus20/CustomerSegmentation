import os
import pandas as pd
import numpy as np

# Load the dataset
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
data_final_path = os.path.join(project_dir, 'data/final/satisfaction_scored_data.csv')
df = pd.read_csv(data_final_path)

# Create a complex churn label based on multiple conditions

# 1. Recency: If RecencyYears is high, the customer is more likely to churn
recency_threshold = 0.3  # Adjust based on data distribution
df['RecencyFlag'] = df['RecencyYears'].apply(lambda x: 1 if x >= recency_threshold else 0)

# 2. Frequency: If TotalPurchases is low, the customer is more likely to churn
frequency_threshold = df['TotalPurchases'].quantile(0.25)  # Bottom 25% in purchases
df['FrequencyFlag'] = df['TotalPurchases'].apply(lambda x: 1 if x <= frequency_threshold else 0)

# 3. Monetary Value: If TotalSpend is low, the customer is more likely to churn
monetary_threshold = df['TotalSpend'].quantile(0.25)  # Bottom 25% in spending
df['MonetaryFlag'] = df['TotalSpend'].apply(lambda x: 1 if x <= monetary_threshold else 0)

# 4. Complaints: If the customer has made complaints, they are more likely to churn
df['ComplainFlag'] = df['Complain'].apply(lambda x: 1 if x > 0 else 0)

# 5. Responses: If the customer has received positive responses, they are less likely to churn
df['ResponseFlag'] = df['Response'].apply(lambda x: 0 if x > 0 else 1)

# 6. Customer Lifetime Value (CLV): If CLV is low, the customer is more likely to churn
clv_threshold = df['CLV'].quantile(0.25)  # Bottom 25% in CLV
df['CLVFlag'] = df['CLV'].apply(lambda x: 1 if x <= clv_threshold else 0)

# 7. Satisfaction Score: If SatisfactionScore is low, the customer is more likely to churn
satisfaction_threshold = df['SatisfactionScore'].quantile(0.25)  # Bottom 25% in satisfaction score
df['SatisfactionFlag'] = df['SatisfactionScore'].apply(lambda x: 1 if x <= satisfaction_threshold else 0)

# Combine the flags to create a complex churn label
# The higher the score, the more likely the customer is to churn
df['ChurnScore'] = (
    df['RecencyFlag'] * 0.15 + 
    df['FrequencyFlag'] * 0.1 + 
    df['MonetaryFlag'] * 0.15 + 
    df['ComplainFlag'] * 0.15 + 
    df['ResponseFlag'] * 0.1 + 
    df['CLVFlag'] * 0.15 +
    df['SatisfactionFlag'] * 0.15
)

# Threshold for churn (you can adjust this based on your data)
churn_threshold = 0.5
df['Churn'] = df['ChurnScore'].apply(lambda x: 1 if x >= churn_threshold else 0)

df = df.drop(columns=['RecencyFlag', 'FrequencyFlag', 'MonetaryFlag', 'ComplainFlag', 'ResponseFlag', 'CLVFlag', 'SatisfactionFlag'])

# Save the updated dataset with churn labels
output_path = os.path.join(project_dir, 'data/final/data_processing_final.csv')
df.to_csv(output_path, index=False)

print(df[['ID', 'ChurnScore', 'Churn']].head(10))  # Print the first 10 rows

print("Complex churn label with satisfaction score created and data saved.")
