import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

# Load the dataset
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
data_final_path = os.path.join(project_dir, 'data/final/data_processing_final.csv')
df_cleaned = pd.read_csv(data_final_path)

# Define features and target
features = ['Income', 'TotalSpend', 'RecencyYears', 'TotalPurchases', 'Complain', 'Response', 'CLV']
X = df_cleaned[features]
y = df_cleaned['SatisfactionScore']

# Handle missing values by imputing with the mean
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Gradient Boosting Regressor
gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)

# Evaluate Models
print("Random Forest MSE:", mean_squared_error(y_test, y_pred_rf))
print("Random Forest R²:", r2_score(y_test, y_pred_rf))

print("Gradient Boosting MSE:", mean_squared_error(y_test, y_pred_gb))
print("Gradient Boosting R²:", r2_score(y_test, y_pred_gb))

# Integrate Predictions
df_cleaned['PredictedSatisfaction_RF'] = rf.predict(imputer.transform(df_cleaned[features]))
df_cleaned['PredictedSatisfaction_GB'] = gb.predict(imputer.transform(df_cleaned[features]))

# Save the results
results_path = os.path.join(project_dir, 'data/final/satisfaction_predictions.csv')
df_cleaned.to_csv(results_path, index=False)

print("Satisfaction prediction completed and data saved.")

