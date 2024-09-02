import pandas as pd
import numpy as np
import random

# Set random seed for reproducibility
np.random.seed(42)

# Number of synthetic customers
num_customers = 500

# Generating random data for each feature
data = {
    'ID': np.arange(1, num_customers + 1),
    'Income': np.random.normal(50000, 15000, num_customers),  # Normal distribution around 50k with 15k std
    'TotalSpend': np.random.normal(3000, 1000, num_customers),  # Normal distribution around 3k with 1k std
    'RecencyYears': np.random.uniform(0.1, 2.5, num_customers),  # Uniform distribution between 0.1 and 2.5 years
    'TotalPurchases': np.random.poisson(10, num_customers),  # Poisson distribution around an average of 10 purchases
    'Complain': np.random.binomial(1, 0.1, num_customers),  # 10% chance of a complaint
    'Response': np.random.binomial(1, 0.3, num_customers),  # 30% chance of response
    'CLV': np.random.normal(1000, 300, num_customers),  # Normal distribution around 1k with 300 std
}

# Creating a DataFrame
df_synthetic = pd.DataFrame(data)

# Ensure all numeric values are positive
df_synthetic['Income'] = df_synthetic['Income'].clip(lower=0)
df_synthetic['TotalSpend'] = df_synthetic['TotalSpend'].clip(lower=0)
df_synthetic['CLV'] = df_synthetic['CLV'].clip(lower=0)

# Saving the synthetic dataset to a CSV file
output_path = 'synthetic_customer_data.csv'
df_synthetic.to_csv(output_path, index=False)

print(f"Synthetic dataset for {num_customers} customers created and saved to {output_path}.")
