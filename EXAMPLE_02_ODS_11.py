# -*- coding: utf-8 -*-
"""
Created on Fri May 24 18:20:33 2024

@author: Jaime
"""
#==============================================================================
# EXAMPLE 02 ODS 11
#==============================================================================
"""
Wang, Lu & Fu (2023) explore the synergy between urban planning and artificial 
intelligence (AI), highlighting how AI improves sustainability, economy and 
quality of life in cities. They discuss ethical and privacy challenges of AI 
in urban environments, highlighting the importance of balancing innovation and 
privacy. They propose to investigate more about the use of AI for land use 
configuration through the analysis of geospatial data and economic and social 
activities.

Wang, Y., Lu, H., & Fu, J. (2023). AI in Urban Planning: Balancing Innovation 
and Privacy. Journal of Urban Development, 12(2), 89-104. 
https://doi.org/10.1016/j.jud.2023.01.005.

This Python code integrates RandomForestRegressor with an evolutionary 
algorithm to optimize land use configuration for urban planning, promoting 
sustainability, economy, and quality of life in cities. The genetic algorithm, 
implemented using the DEAP library, optimizes the hyperparameters of the 
RandomForest model (number of estimators, maximum depth, minimum samples split, 
                    and minimum samples leaf) to minimize the root mean square 
error (RMSE) on the quality of life predictions. The synthetic dataset includes 
various features relevant to urban planning, such as population density, 
income level, employment rate, and access to services. The evolutionary 
algorithm evolves a population of individuals, each representing a set of 
hyperparameters, to find the best combination that optimizes the model's 
performance.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import random
from deap import base, creator, tools, algorithms
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Generate synthetic geospatial and socio-economic data
data = {
    'land_use': np.random.choice(['residential', 'commercial', 'industrial', 'park'], 1000),
    'population_density': np.random.rand(1000) * 1000,
    'income_level': np.random.rand(1000) * 100000,
    'employment_rate': np.random.rand(1000),
    'access_to_services': np.random.rand(1000),
    'quality_of_life': np.random.rand(1000) * 100
}
df = pd.DataFrame(data)

# Encode categorical variables
df = pd.get_dummies(df, columns=['land_use'])

# Split the data into features and target
X = df.drop('quality_of_life', axis=1)
y = df['quality_of_life']

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the evaluation function for the genetic algorithm
def evaluate(individual):
    n_estimators, max_depth, min_samples_split, min_samples_leaf = individual
    model = RandomForestRegressor(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        min_samples_split=int(min_samples_split),
        min_samples_leaf=int(min_samples_leaf),
        random_state=42
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    return (rmse,)

# Set up genetic algorithm tools
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_int", random.randint, 10, 200)
toolbox.register("attr_float", random.uniform, 2, 50)
toolbox.register("individual", tools.initCycle, creator.Individual, 
                 (toolbox.attr_int, toolbox.attr_float, toolbox.attr_float, toolbox.attr_float), n=1)
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

    # Train a neural network on the best hyperparameters
    model = RandomForestRegressor(
        n_estimators=int(best_individual[0]),
        max_depth=int(best_individual[1]),
        min_samples_split=int(best_individual[2]),
        min_samples_leaf=int(best_individual[3]),
        random_state=42
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(f"RMSE of the optimized model: {rmse}")

if __name__ == "__main__":
    main()
