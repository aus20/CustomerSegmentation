import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
data_final_path = os.path.join(project_dir, 'data/final/feature_engineered_data.csv')
df_cleaned = pd.read_csv(data_final_path)

# selecting features for clustering
# extract from df to create X
features = ['Income', 'TotalSpend', 'AvgPurchaseValue', 'CLV', 'RecencyYears', 'AvgFrequencyRate']
X = df_cleaned[features]

# convert any infinite values to NaN
# drop rows with NaN values
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.dropna(inplace=True)

# ensuring df_cleaned matches X in row count after dropping NaNs
df_cleaned = df_cleaned.loc[X.index]

# standartize the data to have mean=0 and variance=1
# because K-Means is sensitive to the scale of the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#divide data into 3 clusters just for a start
#fit_predit() returns the cluster index for each sample
# adds  a new column 'Segment' to the df_cleaned 0 1 2 
kmeans = KMeans(n_clusters=3, random_state=42)
df_cleaned['Segment'] = kmeans.fit_predict(X_scaled)

# PCA reduces the dimensionality(features) of the data to 2 dimensions
# for easy visualization
# fit_transform() returns the reduced data
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
# scatter plot of the reduced data
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df_cleaned['Segment'], palette='viridis')
plt.title('Customer Segments Visualization (PCA)')
# PCA components do not have a direct meaning
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.savefig(os.path.join(project_dir, 'new_reports', 'customer_segments_pca.png'))
plt.show()



# sns.boxplot: Creates a boxplot to visualize the distribution of CLV across different customer segments.
# x='Segment': The x-axis represents the different customer segments.
# y='CLV'
plt.figure(figsize=(10, 6))
sns.boxplot(x='Segment', y='CLV', data=df_cleaned)
plt.title('Customer Segments vs. CLV')
plt.xlabel('Segment')
plt.ylabel('CLV')
plt.savefig(os.path.join(project_dir, 'new_reports', 'clv_segment_boxplot.png'))
plt.show()

# Save the segmented data
segmented_data_path = os.path.join(project_dir, 'data/final/segmented_customer_data.csv')
df_cleaned.to_csv(segmented_data_path, index=False)

print("Clustering completed and data saved.")
