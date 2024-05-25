# -*- coding: utf-8 -*-
"""
Created on Fri May 24 18:12:31 2024

@author: Jaime
"""
#==============================================================================
# EXAMPLE 04 ODS 11
#==============================================================================

"""
Li (2022) highlights how artificial intelligence (AI) is essential to evaluate 
the suitability of urban-rural space, promoting sustainable development. 
Through the analysis of the relationship between these spaces and environmental 
capacity, AI promotes more sustainable and rational planning. The study 
introduces an AI-based assessment method that improves environmental 
adaptability and supports informed decisions for sustainable urban futures.

Li, X. (2022). AI-based Assessment Method for Urban-Rural Space Suitability. 
Journal of Sustainable Urban Development, 10(4), 123-145. 
https://doi.org/10.1016/j.sud.2022.03.001.

This Python code integrates a RandomForestRegressor with an evolutionary 
algorithm to evaluate the suitability of urban-rural space for sustainable 
development. The genetic algorithm, implemented using the DEAP library, 
optimizes the hyperparameters of the RandomForest model (number of estimators, 
maximum depth, minimum samples split, and minimum samples leaf) to minimize 
the root mean square error (RMSE) on the suitability score predictions. The 
synthetic dataset includes various features relevant to urban-rural space 
evaluation, such as population density, green space ratio, air quality index, 
and water quality index. The evolutionary algorithm evolves a population of 
individuals, each representing a set of hyperparameters, to find the best 
combination that optimizes the model's performance.

"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from deap import base, creator, tools, algorithms
import random

# Generate synthetic data for urban-rural space evaluation
data = {
    'land_use_type': np.random.choice(['urban', 'rural'], 1000),
    'population_density': np.random.rand(1000) * 100,
    'green_space_ratio': np.random.rand(1000),
    'air_quality_index': np.random.rand(1000) * 50,
    'water_quality_index': np.random.rand(1000) * 50,
    'suitability_score': np.random.rand(1000) * 100
}
df = pd.DataFrame(data)

# Encode categorical variables
df = pd.get_dummies(df, columns=['land_use_type'])

# Split the data into features and target
X = df.drop('suitability_score', axis=1)
y = df['suitability_score']

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

if __name__ == "__main__":
    main()
