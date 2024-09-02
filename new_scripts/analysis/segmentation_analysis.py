import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np


project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
segmented_data_path = os.path.join(project_dir, 'data/final/satisfaction_predictions.csv')
df_segmented = pd.read_csv(segmented_data_path)

reports_dir = os.path.join(project_dir, 'new_reports')
os.makedirs(reports_dir, exist_ok=True)

# only numeric columns for the summary
numeric_columns = df_segmented.select_dtypes(include=[np.number]).columns
segment_summary = df_segmented.groupby('Segment')[numeric_columns].mean()
print("Segment Summary:\n", segment_summary)

# ave the segment summary to a CSV file
summary_path = os.path.join(reports_dir, 'segment_summary.csv')
segment_summary.to_csv(summary_path)

# Visual Analysis: Income Distribution by Segment
plt.figure(figsize=(10, 6))
sns.boxplot(x='Segment', y='Income', data=df_segmented, palette="Set3")
plt.title('Income Distribution by Segment')
plt.xlabel('Segment')
plt.ylabel('Income')
income_dist_path = os.path.join(reports_dir, 'income_distribution_by_segment.png')
plt.savefig(income_dist_path)
plt.show()

# Visual Analysis: Total Spend by Segment
plt.figure(figsize=(10, 6))
sns.boxplot(x='Segment', y='TotalSpend', data=df_segmented, palette="Set2")
plt.title('Total Spend by Segment')
plt.xlabel('Segment')
plt.ylabel('Total Spend')
spend_dist_path = os.path.join(reports_dir, 'spend_distribution_by_segment.png')
plt.savefig(spend_dist_path)
plt.show()

# Visual Analysis: CLV Distribution by Segment
plt.figure(figsize=(10, 6))
sns.boxplot(x='Segment', y='CLV', data=df_segmented, palette="Set1")
plt.title('CLV Distribution by Segment')
plt.xlabel('Segment')
plt.ylabel('CLV')
clv_dist_path = os.path.join(reports_dir, 'clv_distribution_by_segment.png')
plt.savefig(clv_dist_path)
plt.show()

# Visual Analysis: Satisfaction Score by Segment
plt.figure(figsize=(10, 6))
sns.boxplot(x='Segment', y='SatisfactionScore', data=df_segmented, palette="Set3")
plt.title('Satisfaction Score by Segment')
plt.xlabel('Segment')
plt.ylabel('Satisfaction Score')
satisfaction_dist_path = os.path.join(reports_dir, 'satisfaction_distribution_by_segment.png')
plt.savefig(satisfaction_dist_path)
plt.show()

# Calculate percentages for each Education level within each Segment
education_segment = df_segmented.groupby(['Segment', 'Education']).size().unstack(fill_value=0)
education_segment_percent = education_segment.div(education_segment.sum(axis=1), axis=0)

# Plotting
education_segment_percent.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='Set3')
plt.title('Education Level vs. Segment (Percentage)')
plt.xlabel('Segment')
plt.ylabel('Percentage')
plt.legend(title='Education Level', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig(os.path.join(project_dir, reports_dir, 'education_vs_segment_percentage.png'))
plt.show()

# Family Size vs. Segment
plt.figure(figsize=(10, 6))
sns.boxplot(x='Segment', y='Family_Size', data=df_segmented, palette="Set2")
plt.title('Family Size vs. Segment')
plt.xlabel('Segment')
plt.ylabel('Family Size')
plt.savefig(os.path.join(project_dir, reports_dir, 'family_size_vs_segment.png'))
plt.show()

# Correlation Analysis within Segments

for segment in df_segmented['Segment'].unique():
    plt.figure(figsize=(12, 10))
    segment_data = df_segmented[df_segmented['Segment'] == segment]
    
    # Filter only numeric columns
    numeric_data = segment_data.select_dtypes(include=[np.number])
    
    sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title(f'Correlation Heatmap for Segment {segment}')
    correlation_path = os.path.join(reports_dir, f'correlation_segment_{segment}.png')
    plt.savefig(correlation_path)
    plt.show()

# Save Segment-Specific Data
for segment in df_segmented['Segment'].unique():
    segment_data = df_segmented[df_segmented['Segment'] == segment]
    segment_data_path = os.path.join(reports_dir, f'segment_{segment}_data.csv')
    segment_data.to_csv(segment_data_path, index=False)

print("Detailed segment analysis completed and saved.")


