# -*- coding: utf-8 -*-
"""
Created on Wed May 29 12:51:45 2024

@author: Jaime
"""

"""
Yigitcanlar et. al (2020) review the impact of artificial intelligence (AI) on 
the development of smart cities, addressing sectors such as business, 
education, energy, health, and transportation. The study highlights how AI 
influences urban sustainability, its disruptive benefits, and challenges and 
opportunities for future research. They highlight the importance of analyzing 
the potential risks and disruptions of AI in urban planning. 

Yiğitcanlar, T., Desouza, K. C., Butler, L., & Roozkhosh, F. (2020). 
Contributions and Risks of  Artificial Intelligence (AI) in Building Smarter 
Cities: Insights from a Systematic Review of the Literature. Energies, 13(6), 
1473. https://doi.org/10.3390/en13061473.

The provided Python code implements an algorithm to analyze the impact of 
artificial intelligence (AI) on the development of smart cities by clustering 
cities based on various sectoral indicators (business, education, energy, 
                                             health, and transportation). 
The dataset, assumed to contain relevant sectoral information, is first 
normalized to ensure equal contribution from all features. KMeans clustering 
is then applied to group the cities into clusters with similar characteristics. 
The algorithm further analyzes and visualizes the cluster centers to 
understand the common traits within each cluster. Additionally, it highlights 
potential risks and opportunities for AI in urban planning by examining the 
cluster characteristics, saving the analysis results for further examination.
"""

#==============================================================================
# EXAMPEL 10 ODS 11
#==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset
# This dataset should contain information about different sectors such as business, education, energy, health, and transportation
# For example, each row might represent a city and columns might represent different indicators for these sectors
# Example: dataset = pd.read_csv('smart_city_data.csv')

# Comment: Replace the placeholder with the actual dataset path
dataset_path = 'smart_city_data.csv'
dataset = pd.read_csv(dataset_path)

# Preprocessing the data
# Comment: Normalize the data to ensure all features contribute equally to the analysis
scaler = StandardScaler()
scaled_data = scaler.fit_transform(dataset)

# Define the number of clusters for KMeans
# Comment: The number of clusters can be tuned based on the specific requirements or using methods like the elbow method
num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=42)

# Fit the KMeans algorithm to the data
kmeans.fit(scaled_data)

# Assign each city to a cluster
dataset['Cluster'] = kmeans.labels_

# Analyze the results
# Comment: Group data by clusters to analyze the common characteristics of each cluster
cluster_analysis = dataset.groupby('Cluster').mean()

# Visualize the cluster centers
# Comment: Plot the cluster centers to understand the characteristics of each cluster
plt.figure(figsize=(10, 6))
for i in range(num_clusters):
    plt.plot(kmeans.cluster_centers_[i], label=f'Cluster {i}')

plt.title('Cluster Centers')
plt.xlabel('Feature Index')
plt.ylabel('Cluster Center Value')
plt.legend()
plt.show()

# Comment: Save the analysis results to a CSV file for further examination
cluster_analysis.to_csv('cluster_analysis_results.csv')

# Comment: Highlight potential risks and opportunities for AI in urban planning based on cluster characteristics
# Here, add specific domain knowledge about how different sectors (business, education, energy, health, transportation) interact with AI
# Example: Risks in energy might involve over-reliance on AI leading to vulnerabilities in power grids

# Comment: Example function to analyze risks and opportunities
def analyze_ai_impact(cluster_data):
    risks = {}
    opportunities = {}
    for cluster, data in cluster_data.iterrows():
        # Example risk analysis for energy sector
        if data['energy'] > threshold:
            risks[cluster] = 'High dependency on AI in energy sector'
        # Example opportunity analysis for education sector
        if data['education'] < threshold:
            opportunities[cluster] = 'AI can improve educational outcomes'
    return risks, opportunities

# Define a threshold for example purposes
threshold = 0.5

# Perform the analysis
risks, opportunities = analyze_ai_impact(cluster_analysis)

# Save the risks and opportunities analysis to a file
with open('ai_impact_analysis.txt', 'w') as f:
    f.write("Risks:\n")
    for cluster, risk in risks.items():
        f.write(f"Cluster {cluster}: {risk}\n")
    f.write("\nOpportunities:\n")
    for cluster, opportunity in opportunities.items():
        f.write(f"Cluster {cluster}: {opportunity}\n")
