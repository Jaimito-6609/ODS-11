# -*- coding: utf-8 -*-
"""
Created on Fri May 24 18:37:00 2024

@author: Jaime
"""
#==============================================================================
# EXAMPLE 06 ODS 11
#==============================================================================
"""
Abid et. al (2021) highlight the essentiality of artificial intelligence (AI) 
in public safety and emergency response. Using techniques from data analysis, 
machine learning, tracking, geospatial analysis and robotics, AI improves 
disaster management, contributing to urban resilience and safety. This study 
emphasizes the importance of integrating geographic information systems and 
remote sensing in all stages of disaster management.

Abid, S. K., Sulaiman, N., Chan, S. W., Nazir, U., Abid, M., Han, H., 
Ariza-Montes, A., & Vega-Muñoz, A. (2021). Toward an Integrated Disaster 
Management Approach: How Artificial Intelligence Can Boost Disaster 
Management. Sustainability, 13(22), 12560. https://doi.org/10.3390/su132212560.

This Python code integrates Random Forest classifiers with a genetic algorithm 
to optimize disaster management and improve public safety and emergency 
response. The genetic algorithm, implemented using the DEAP library, optimizes 
the hyperparameters of the Random Forest model (number of estimators, 
maximum depth, minimum samples split, and minimum samples leaf) to maximize 
the accuracy of disaster severity predictions. The synthetic dataset includes 
various features relevant to disaster management, such as earthquake risk, 
flood risk, fire risk, and emergency response time. The evolutionary algorithm 
evolves a population of individuals, each representing a set of hyperparameters, 
to find the best combination that optimizes the model's performance. This 
approach emphasizes the integration of geographic information systems (GIS) 
and remote sensing in all stages of disaster management, contributing to urban 
resilience and safety.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import random
from deap import base, creator, tools, algorithms
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam

# Generate synthetic data for disaster management
data = {
    'location': [f"loc_{i}" for i in range(1000)],
    'latitude': np.random.uniform(-90, 90, 1000),
    'longitude': np.random.uniform(-180, 180, 1000),
    'earthquake_risk': np.random.rand(1000),
    'flood_risk': np.random.rand(1000),
    'fire_risk': np.random.rand(1000),
    'emergency_response_time': np.random.rand(1000) * 100,
    'disaster_severity': np.random.choice([0, 1], 1000)  # 0: low, 1: high
}
df = pd.DataFrame(data)

# Convert to GeoDataFrame
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude))

# Split the data into features and target
X = df.drop(['location', 'disaster_severity', 'geometry'], axis=1)
y = df['disaster_severity']

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the evaluation function for the genetic algorithm
def evaluate(individual):
    n_estimators, max_depth, min_samples_split, min_samples_leaf = individual

    # Build the Random Forest model
    model = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        min_samples_split=int(min_samples_split),
        min_samples_leaf=int(min_samples_leaf),
        random_state=42
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return (1 - accuracy,)

# Set up genetic algorithm tools
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_int", random.randint, 10, 200)
toolbox.register("attr_float", random.uniform, 2, 50)
toolbox.register("individual", tools.initCycle, creator.Individual, 
                 (toolbox.attr_int, toolbox.attr_int, toolbox.attr_float, toolbox.attr_float), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=10, up=200, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

# Main algorithm function
def main():
    population = toolbox.population(n=50)
    ngen = 40
    cxpb = 0.5
    mutpb = 0.2

    # Run the genetic algorithm
    algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngen, 
                        stats=None, halloffame=None, verbose=True)

    # Extract the best individual from the final population
    best_individual = tools.selBest(population, 1)[0]
    print("Best individual is: %s\nwith fitness: %s" % (best_individual, best_individual.fitness.values))

if __name__ == "__main__":
    main()
