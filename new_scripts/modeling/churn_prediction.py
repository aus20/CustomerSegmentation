import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load the dataset
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) 
data_final_path = os.path.join(project_dir, 'data/final/satisfaction_predictions.csv')
df_cleaned = pd.read_csv(data_final_path)

# Feature Engineering
df_cleaned['SpendRatio'] = df_cleaned['TotalSpend'] / (df_cleaned['Income'] + 1)
df_cleaned['RecencyTrend'] = df_cleaned['RecencyYears'].diff().fillna(0)
df_cleaned['EngagementRatio'] = df_cleaned['Response'] / (df_cleaned['TotalPurchases'] + 1)

# Define features and target
features = ['Income', 'TotalSpend', 'RecencyYears', 'TotalPurchases', 'Complain', 'Response', 
            'CLV', 'SpendRatio', 'RecencyTrend', 'EngagementRatio']
X = df_cleaned[features]
y = df_cleaned['Churn']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Model 2: Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)


print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))
print("Random Forest AUC-ROC:", roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]))

# Save model predictions
df_cleaned['PredictedChurn_RF'] = rf.predict_proba(scaler.transform(df_cleaned[features]))[:, 1]

# Save the results
results_path = os.path.join(project_dir, 'data/final/churn_predictions.csv')
df_cleaned.to_csv(results_path, index=False)

print("Churn prediction completed and data saved.")

