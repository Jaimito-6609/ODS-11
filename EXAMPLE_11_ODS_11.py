# -*- coding: utf-8 -*-
"""
Created on Wed May 29 12:59:22 2024

@author: Jaime
"""
"""
Acharyya et. al (2023) highlight the role of prompt engineering in sustainable 
urban development, using artificial intelligence (AI), Internet of Things (IoT) 
and data analysis to improve the management of infrastructure and 
transportation systems. Through the study of concepts, methodologies and 
practical cases, they highlight the need to adopt innovative approaches to 
face urban challenges, aiming for a more sustainable and efficient future. 

Acharyya, S., Mukherjee, M. S., Saha, S., & Pal, D. (2023). Revolutionizing 
Natural Language Understanding with Prompt Engineering: A Comprehensive Study. 
International Research Journal Of Innovations In Engineering And Technology, 
07(10), 692-695. https://doi.org/10.47001/irjiet/2023.710091.

The provided Python code implements an algorithm to support sustainable urban 
development by leveraging AI, IoT, and data analysis. The dataset, assumed to 
contain relevant urban development indicators, is processed to select key 
features and define the target variable, which is the sustainability score. 
The features are normalized, and the data is split into training and testing 
sets. A RandomForestRegressor model is trained on the training data and 
evaluated on the test data. The model, along with the scaler, is saved for 
future use. Additionally, a function is defined to generate prompts for AI 
and IoT integration in urban management, aiming to improve infrastructure and 
transportation systems. The generated prompts are saved for further use.
"""

#==============================================================================
# EXAMPEL 11 ODS 11
#==============================================================================
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib

# Load the dataset
# The dataset should contain information on infrastructure and transportation systems influenced by AI, IoT, and data analysis
# Example: dataset = pd.read_csv('urban_development_data.csv')

# Placeholder for the dataset path
dataset_path = 'urban_development_data.csv'
dataset = pd.read_csv(dataset_path)

# Feature selection and target definition
# Comment: Select relevant features for modeling and define the target variable
features = ['infrastructure_index', 'transportation_index', 'AI_integration_level', 'IoT_coverage', 'data_quality']
target = 'sustainability_score'

X = dataset[features]
y = dataset[target]

# Preprocessing the data
# Comment: Normalize the features to ensure consistent scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and train the RandomForest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Save the model and scaler for future use
joblib.dump(model, 'urban_dev_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Prompt engineering for AI and IoT integration
# Comment: Define a function to generate prompts for AI and IoT systems to improve urban infrastructure and transportation management
def generate_prompts(city_data):
    prompts = []
    for index, row in city_data.iterrows():
        prompt = f"City {row['city_name']}: Integrate AI at {row['AI_integration_level']} level and expand IoT coverage to {row['IoT_coverage']} to enhance infrastructure and transportation management."
        prompts.append(prompt)
    return prompts

# Example usage of prompt generation
prompts = generate_prompts(dataset)
for prompt in prompts[:5]:
    print(prompt)

# Save the generated prompts
with open('ai_iot_prompts.txt', 'w') as f:
    for prompt in prompts:
        f.write(f"{prompt}\n")
