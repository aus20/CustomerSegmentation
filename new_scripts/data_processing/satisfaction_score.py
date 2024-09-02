import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load the cleaned dataset
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
data_final_path = os.path.join(project_dir, 'data/final/segmented_customer_data.csv')
df_cleaned = pd.read_csv(data_final_path)

# Define refined weights for features based on their importance
weights = {
    'TotalSpend': 0.2,
    'RecencyYears': -0.25,
    'TotalPurchases': 0.3,
    'Complain': -0.35,
    'Response': 0.3  
}
# Normalize weights so they sum to 1
weight_sum = sum(abs(v) for v in weights.values())
normalized_weights = {k: v / weight_sum for k, v in weights.items()}

# Calculate the weighted scores for each feature
df_cleaned['TotalSpendWeighted'] = df_cleaned['TotalSpend'] * normalized_weights['TotalSpend']
df_cleaned['RecencyWeighted'] = df_cleaned['RecencyYears'] * normalized_weights['RecencyYears']
df_cleaned['TotalPurchasesWeighted'] = df_cleaned['TotalPurchases'] * normalized_weights['TotalPurchases']
df_cleaned['ComplainWeighted'] = df_cleaned['Complain'] * normalized_weights['Complain']
df_cleaned['ResponseWeighted'] = df_cleaned['Response'] * normalized_weights['Response']

# Combine all weighted features into a new DataFrame
weighted_features = df_cleaned[['TotalSpendWeighted', 'RecencyWeighted', 'TotalPurchasesWeighted', 'ComplainWeighted', 'ResponseWeighted']]

# Apply Min-Max Scaling to each feature to normalize them between 0 and 1
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(weighted_features)

# Sum the scaled features to create the final SatisfactionScore
df_cleaned['SatisfactionScore'] = np.sum(scaled_features, axis=1)

# Normalize SatisfactionScore between 0 and 10 for readability
df_cleaned['SatisfactionScore'] = (df_cleaned['SatisfactionScore'] - df_cleaned['SatisfactionScore'].min()) / (df_cleaned['SatisfactionScore'].max() - df_cleaned['SatisfactionScore'].min()) * 10

# Delete the weighted feature columns
df_cleaned = df_cleaned.drop(columns=['TotalSpendWeighted', 'RecencyWeighted', 'TotalPurchasesWeighted', 'ComplainWeighted', 'ResponseWeighted'])

# Save the updated dataset
satisfaction_data_path = os.path.join(project_dir, 'data/final/satisfaction_scored_data.csv')
df_cleaned.to_csv(satisfaction_data_path, index=False)

# Print the satisfaction scores for inspection
print(df_cleaned[['ID', 'SatisfactionScore']])



