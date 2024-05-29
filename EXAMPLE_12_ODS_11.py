# -*- coding: utf-8 -*-
"""
Created on Wed May 29 13:05:52 2024

@author: Jaime
"""
"""
Youssef (2021) examines the importance of implementing smart city indicators 
in urban planning to address accelerated urban growth and sustainability. He 
highlights the interaction between artificial intelligence (AI), technology 
and sustainability to transform cities into smart ones through key performance 
indicators. This approach addresses problems such as traffic, sprawl, and loss 
of natural resources, focusing on preserving urban efficiency and 
sustainability. Youssef, T. A. (2021). SMART CITIES AS a SUSTAINABLE 
DEVELOPMENT TREND AND URBAN GROWTH CHALLENGES. Journal Of Al-Azhar University 
Engineering Sector, 16(58), 203-222. 
https://doi.org/10.21608/auej.2021.141775." 

The provided Python code implements an algorithm to analyze the importance of 
implementing smart city indicators in urban planning for addressing urban 
growth and sustainability challenges. The dataset, which includes various 
smart city indicators, is first normalized to ensure consistency. Principal 
Component Analysis (PCA) is applied to reduce the dimensionality of the data 
while retaining most of the variance. KMeans clustering is then used to 
identify patterns in the smart city indicators. The optimal number of clusters 
is determined using the elbow method. The data is clustered, and each data 
point is assigned to a cluster. The clusters are analyzed to identify key 
patterns, and the results are visualized using the first two principal 
components. Finally, the algorithm generates recommendations for urban 
planning based on the characteristics of each cluster, focusing on improving 
AI integration, addressing traffic issues, managing urban sprawl, and 
conserving natural resources.
"""

#==============================================================================
# EXAMPEL 12 ODS 11
#==============================================================================
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the dataset
# The dataset should contain smart city indicators relevant to urban planning, sustainability, and growth challenges
# Example: dataset = pd.read_csv('smart_city_indicators.csv')

# Placeholder for the dataset path
dataset_path = 'smart_city_indicators.csv'
dataset = pd.read_csv(dataset_path)

# Preprocessing the data
# Comment: Normalize the data to ensure consistency across different indicators
scaler = StandardScaler()
scaled_data = scaler.fit_transform(dataset)

# Applying PCA for dimensionality reduction
# Comment: Use PCA to reduce the dimensionality of the data while retaining most of the variance
pca = PCA(n_components=0.95)  # Retain 95% of variance
pca_data = pca.fit_transform(scaled_data)

# KMeans clustering to identify patterns in smart city indicators
# Comment: Determine the optimal number of clusters using the elbow method
def find_optimal_clusters(data, max_k):
    iters = range(2, max_k+1, 1)
    sse = []
    for k in iters:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        sse.append(kmeans.inertia_)
    plt.figure(figsize=(10, 6))
    plt.plot(iters, sse, marker='o')
    plt.xlabel('Cluster Centers')
    plt.ylabel('SSE')
    plt.title('Elbow Method For Optimal k')
    plt.show()

# Find the optimal number of clusters
find_optimal_clusters(pca_data, 10)

# Define the number of clusters based on the elbow method
num_clusters = 4  # This value should be chosen based on the elbow plot
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(pca_data)

# Assign each data point to a cluster
dataset['Cluster'] = kmeans.labels_

# Analyze the clusters to identify key patterns in smart city indicators
cluster_analysis = dataset.groupby('Cluster').mean()

# Visualize the clusters using the first two principal components
plt.figure(figsize=(10, 6))
for i in range(num_clusters):
    plt.scatter(pca_data[kmeans.labels_ == i, 0], pca_data[kmeans.labels_ == i, 1], label=f'Cluster {i}')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Clusters of Smart City Indicators')
plt.legend()
plt.show()

# Save the cluster analysis results
cluster_analysis.to_csv('cluster_analysis_smart_city.csv')

# Generate recommendations for urban planning based on cluster characteristics
# Comment: Define a function to generate recommendations for each cluster
def generate_recommendations(cluster_data):
    recommendations = {}
    for cluster, data in cluster_data.iterrows():
        recommendation = f"Cluster {cluster}: Focus on improving AI integration and sustainable practices in urban planning. Address traffic issues, manage urban sprawl, and conserve natural resources."
        recommendations[cluster] = recommendation
    return recommendations

# Generate recommendations
recommendations = generate_recommendations(cluster_analysis)

# Save the recommendations
with open('urban_planning_recommendations.txt', 'w') as f:
    for cluster, recommendation in recommendations.items():
        f.write(f"{recommendation}\n")
