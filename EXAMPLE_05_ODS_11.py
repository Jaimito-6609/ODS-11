# -*- coding: utf-8 -*-
"""
Created on Fri May 24 18:30:44 2024

@author: Jaime
"""
#==============================================================================
# EXAMPLE 05 ODS 11
#==============================================================================
"""
Cisse (2023) emphasizes how artificial intelligence (AI) is transforming 
environmental monitoring and air quality management through deep data 
collection and analysis. This approach allows for more effective measures to 
be implemented in cities, improving public health and urban sustainability. 
Highlights the relevance of adaptive urban planning and data integration in 
urban contexts, especially in Africa.

Cissé, C. (2023). At the crossroads of datas: Using artificial intelligence 
for efficient and resilient planning in African cities. Urban Resilience And 
Sustainability, 1(4), 309-313. https://doi.org/10.3934/urs.2023019.

This Python code integrates LSTM neural networks with a genetic algorithm to 
optimize air quality monitoring and management in urban environments. The 
genetic algorithm, implemented using the DEAP library, optimizes the 
hyperparameters of the LSTM neural network (epochs, batch size, and LSTM units) 
to minimize the root mean square error (RMSE) on the air quality index 
predictions. The synthetic dataset includes various features relevant to air 
quality, such as PM2.5, PM10, NO2, CO concentrations, temperature, and 
humidity. The evolutionary algorithm evolves a population of individuals, 
each representing a set of hyperparameters, to find the best combination 
that optimizes the LSTM model's performance. This approach aims to improve 
public health and urban sustainability through adaptive urban planning and 
data integration, especially in African cities.
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

# Generate synthetic data for air quality monitoring
data = {
    'pm2_5': np.random.rand(1000) * 100,  # PM2.5 concentration
    'pm10': np.random.rand(1000) * 100,   # PM10 concentration
    'no2': np.random.rand(1000) * 100,    # NO2 concentration
    'co': np.random.rand(1000) * 10,      # CO concentration
    'temperature': np.random.rand(1000) * 40,  # Temperature in Celsius
    'humidity': np.random.rand(1000) * 100,    # Humidity in percentage
    'air_quality_index': np.random.rand(1000) * 500  # Air Quality Index (AQI)
}
df = pd.DataFrame(data)

# Split the data into features and target
X = df.drop('air_quality_index', axis=1)
y = df['air_quality_index']

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
toolbox.register("attr_float", random.uniform, 1, 50)
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
