# -*- coding: utf-8 -*-
"""
Created on Wed May 29 13:12:45 2024

@author: Jaime
"""
"""
Hsu et. al (2023) study how artificial intelligence (AI) and the Internet of 
Things (IoT) can revolutionize environmental governance in Chinese cities. 
The study highlights the role of digitalization in closing data gaps, improving 
policy analysis and transforming government and social interactions. Analyzes 
smart city initiatives and discusses the opportunities and challenges of 
implementing digital environmental management, highlighting the improvement of 
analytical capacity and the generation of new data. Hsu, A., Li, L., Schletz, 
M., & Yu, Z. (2023). Chinese cities as digital environmental governance 
innovators: Evidence from subnational low-Carbon plans. Environment And 
lanning. B, Urban Analytics And City Science/Environment & Planning. B, Urban 
Analytics And City Science, 51(3), 572-589. 
https://doi.org/10.1177/23998083231186622.

The provided Python code implements an algorithm to analyze the role of AI and 
IoT in revolutionizing environmental governance in Chinese cities. The dataset, 
assumed to contain relevant environmental governance indicators, is processed 
to select key features and define the target variable, which is the 
environmental performance class. The features are normalized, and the data is 
split into training and testing sets. A RandomForestClassifier model is 
trained on the training data and evaluated on the test data. The model, along 
with the scaler, is saved for future use. Additionally, a function is defined 
to generate insights and recommendations for improving environmental 
governance, focusing on enhancing digitalization, increasing IoT 
implementation, closing data gaps, and adopting AI for better policy 
effectiveness and environmental performance. The generated insights are saved 
for further use.
"""

#==============================================================================
# EXAMPEL 13 ODS 11
#==============================================================================
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Load the dataset
# The dataset should contain information on environmental governance indicators influenced by AI and IoT in Chinese cities
# Example: dataset = pd.read_csv('environmental_governance_data.csv')

# Placeholder for the dataset path
dataset_path = 'environmental_governance_data.csv'
dataset = pd.read_csv(dataset_path)

# Feature selection and target definition
# Comment: Select relevant features for modeling and define the target variable
features = ['digitalization_level', 'policy_effectiveness', 'data_availability', 'AI_adoption', 'IoT_implementation']
target = 'environmental_performance_class'

X = dataset[features]
y = dataset[target]

# Preprocessing the data
# Comment: Normalize the features to ensure consistent scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and train the RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(classification_report(y_test, y_pred))

# Save the model and scaler for future use
joblib.dump(model, 'environmental_governance_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Generate insights for digital environmental management
# Comment: Define a function to generate insights and recommendations for improving environmental governance
def generate_insights(city_data):
    insights = []
    for index, row in city_data.iterrows():
        insight = f"City {row['city_name']}: Enhance digitalization and increase IoT implementation to improve policy effectiveness and environmental performance. Focus on closing data gaps and adopting AI for better governance."
        insights.append(insight)
    return insights

# Example usage of insight generation
insights = generate_insights(dataset)
for insight in insights[:5]:
    print(insight)

# Save the generated insights
with open('digital_environmental_management_insights.txt', 'w') as f:
    for insight in insights:
        f.write(f"{insight}\n")
