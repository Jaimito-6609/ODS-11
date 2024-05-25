# -*- coding: utf-8 -*-
"""
Created on Fri May 24 18:48:47 2024

@author: Jaime
"""
#==============================================================================
# EXAMPLE 07 ODS 11
#==============================================================================
"""
Yu et. al (2023) emphasize the importance of artificial intelligence (AI) in 
urban infrastructure development, using AI models, especially improved LSTM, 
to optimize Urban Green Space (UGS) planning in urban areas. These models, 
superior in accuracy, demonstrate how diverse configurations of plant 
communities can reduce PM 2.5 concentrations, promoting a healthier urban 
environment and effective green planning strategies.

Yu, S., Guan, X., Zhu, J., Wang, Z., Jian, Y., Wang, W., & Yang, Y. (2023). 
Artificial Intelligence and Urban Green Space Facilities Optimization Using 
the LSTM Model: Evidence from China. Sustainability, 15(11), 8968. 
https://doi.org/10.3390/su15118968.

This Python code integrates LSTM neural networks with a genetic algorithm to 
optimize Urban Green Space (UGS) planning, aiming to reduce PM 2.5 
concentrations and promote a healthier urban environment. The genetic 
algorithm, implemented using the DEAP library, optimizes the hyperparameters 
of the LSTM neural network (epochs, batch size, and LSTM units) to minimize 
the root mean square error (RMSE) on the PM 2.5 reduction predictions. The 
synthetic dataset includes various features relevant to UGS planning, such as 
plant diversity index, green space area, and average tree height. The 
evolutionary algorithm evolves a population of individuals, each representing 
a set of hyperparameters, to find the best combination that optimizes the 
LSTM model's performance. This approach demonstrates the effectiveness of AI 
in urban infrastructure development and green planning strategies.
"""

import numpy as np
import pandas as pd
import random
from deap import base, creator, tools, algorithms
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam

# Generate synthetic data for Urban Green Space (UGS) planning
data = {
    'plant_diversity_index': np.random.rand(1000) * 10,  # Index of plant diversity
    'green_space_area': np.random.rand(1000) * 100,      # Area of green space in square meters
    'average_tree_height': np.random.rand(1000) * 20,    # Average height of trees in meters
    'pm2_5_reduction': np.random.rand(1000) * 50         # Reduction in PM 2.5 concentrations
}
df = pd.DataFrame(data)

# Split the data into features and target
X = df.drop('pm2_5_reduction', axis=1)
y = df['pm2_5_reduction']

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the evaluation function for the genetic algorithm
def evaluate(individual):
    epochs, batch_size, lstm_units = individual

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units=int(lstm_units), input_shape=(X_train.shape[1], 1)))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer=Adam(), loss='mean_squared_error')

    # Reshape the training data
    X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Train the model
    model.fit(X_train_reshaped, y_train, epochs=int(epochs), batch_size=int(batch_size), verbose=0)

    # Evaluate the model on the test data
    predictions = model.predict(X_test_reshaped)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    return (rmse,)

# Set up genetic algorithm tools
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_int", random.randint, 10, 200)
toolbox.register("attr_float", random.uniform, 10, 100)
toolbox.register("individual", tools.initCycle, creator.Individual, 
                 (toolbox.attr_int, toolbox.attr_int, toolbox.attr_float), n=1)
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
