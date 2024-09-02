import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# better visibility of the plots
sns.set_theme(style="whitegrid")

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
data_cleaned_path = os.path.join(project_dir, 'data/final/feature_engineered_data.csv')
df_cleaned = pd.read_csv(data_cleaned_path)

reports_dir = os.path.join(project_dir, 'new_reports')

# Visualize Spending Patterns
plt.figure(figsize=(12, 6))
sns.histplot(df_cleaned['TotalSpend'], bins=30, kde=True, color="skyblue")
plt.title('Total Spend Distribution', fontsize=18)
plt.xlabel('Total Spend', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.savefig(os.path.join(reports_dir, 'total_spend_distribution.png'))
plt.show()

# Visualize Income Distribution
plt.figure(figsize=(12, 6))
sns.histplot(df_cleaned['Income'], bins=30, kde=True, color="salmon")
plt.title('Income Distribution', fontsize=18)
plt.xlabel('Income', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.savefig(os.path.join(reports_dir, 'income_distribution.png'))
plt.show()

# Visualize Customer Segmentation by Age Group
plt.figure(figsize=(10, 6))
sns.countplot(x='AgeGroup', data=df_cleaned, palette="coolwarm")
plt.title('Age Group Distribution', fontsize=18)
plt.xlabel('Age Group', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig(os.path.join(reports_dir, 'age_group_distribution.png'))
plt.show()

# Correlations between numerical features
numeric_df = df_cleaned.select_dtypes(include=[np.number])
plt.figure(figsize=(24, 20))  # Increase the size for better readability
sns.heatmap(
    numeric_df.corr(), 
    annot=True,          
    fmt='.2f',            
    cmap='coolwarm',      
    cbar=True,            
    square=True,          
    linewidths=0.5,      
    linecolor='white',    
    annot_kws={"size": 10}  
)
plt.title('Correlation Heatmap', fontsize=24)  
plt.xticks(fontsize=12) 
plt.yticks(fontsize=12)  
plt.tight_layout()       
plt.savefig(os.path.join(reports_dir, 'correlation_heatmap.png'))
plt.show()

# Outlier Detection in CLV
plt.figure(figsize=(10, 6))
sns.boxplot(y='CLV', data=df_cleaned, color="lightcoral")
plt.title('Outlier Detection in CLV', fontsize=18)
plt.ylabel('CLV', fontsize=14)
plt.yticks(fontsize=12)
plt.savefig(os.path.join(reports_dir, 'clv_outliers.png'))
plt.show()

print("Exploratory Data Analysis completed and reports saved.")
