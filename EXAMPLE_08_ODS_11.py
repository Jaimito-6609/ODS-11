# -*- coding: utf-8 -*-
"""
Created on Tue May 28 18:39:34 2024

@author: Jaime
"""

"""
Deep (2023) highlights how artificial intelligence (AI) and smart technologies 
are transforming urban planning, promoting social inclusion and affordable 
housing through real estate market analysis and equitable policies. However, 
it warns of challenges such as data privacy and cyber threats, underscoring 
the need for ethical implementation. These technologies, such as transportation 
and smart networks, together with digital sensors, optimize the management of 
urban resources and quality of life, requiring a balance between innovation and 
ethics.

Deep, G. (2023). The impact of technology on Urban infrastructure. 
International Journal Of Science And Research Archive, 10(2), 664-668. 
https://doi.org/10.30574/ijsra.2023.10.2.0995.

This Python code leverages RandomForestRegressor with GridSearchCV to optimize 
parameters for real estate market analysis, aiming to promote social inclusion 
and affordable housing through urban planning. The synthetic dataset includes 
various features relevant to property market value, such as property size, 
number of bedrooms and bathrooms, location score, and age of the property. The 
GridSearchCV is used to find the best hyperparameters for the Random Forest 
model (number of estimators, maximum depth, minimum samples split, and minimum 
       samples leaf). The model's performance is evaluated using Mean Squared 
Error (MSE), R^2 Score, and Mean Absolute Error (MAE). This approach 
demonstrates how AI and smart technologies can optimize the management of 
urban resources and quality of life while balancing innovation and ethics.
"""

import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Generate synthetic data for real estate market analysis
data = {
    'property_size': np.random.rand(1000) * 100,  # Property size in square meters
    'num_bedrooms': np.random.randint(1, 5, 1000),  # Number of bedrooms
    'num_bathrooms': np.random.randint(1, 3, 1000),  # Number of bathrooms
    'location_score': np.random.rand(1000) * 10,  # Location score based on amenities and transportation
    'age_of_property': np.random.rand(1000) * 50,  # Age of the property in years
    'market_value': np.random.rand(1000) * 500000  # Market value of the property in USD
}
df = pd.DataFrame(data)

# Split the data into features and target
X = df.drop('market_value', axis=1)
y = df['market_value']

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize the Random Forest Regressor
rf = RandomForestRegressor(random_state=42)

# Initialize GridSearchCV with the Random Forest Regressor and the parameter grid
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

# Fit the model to find the best parameters
grid_search.fit(X_train, y_train)

# Extract the best model
best_rf = grid_search.best_estimator_

# Evaluate the best model on the test data
y_pred = best_rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Print the evaluation metrics
print(f"Best Random Forest parameters: {grid_search.best_params_}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R^2 Score: {r2}")
print(f"Mean Absolute Error (MAE): {mae}")
