# -*- coding: utf-8 -*-
"""
Created on Tue May 28 18:46:01 2024

@author: Jaime
"""

"""
Russo and Comi (2023) investigate the ethical and privacy challenges of 
artificial intelligence (AI) in urban planning, highlighting the need for 
critical reflection on the use of data and surveillance. Their study proposes 
a two-layer route optimization model that integrates IoT, big data, blockchain 
and AI, combining historical and real-time data for urban messaging. They 
conclude by emphasizing innovations and cost optimization in urban deliveries.

Russo, F., & Comi, A. (2023). Urban Courier Delivery in a Smart City: The User 
Learning Process of Travel Costs Enhanced by Emerging Technologies. 
Sustainability, 15(23), 16253. https://doi.org/10.3390/su152316253.

This Python code integrates LSTM neural networks with a genetic algorithm to 
optimize urban delivery routes, focusing on cost optimization and efficiency 
in a smart city environment. The synthetic dataset includes various features 
relevant to delivery cost prediction, such as delivery distance, delivery time, 
traffic conditions, number of stops, and historical delivery cost. The genetic 
algorithm, implemented using the DEAP library, optimizes the hyperparameters 
of the LSTM neural network (epochs, batch size, and LSTM units) to minimize the 
root mean square error (RMSE) on the real-time delivery cost predictions. This 
approach combines IoT, big data, blockchain, and AI to enhance urban messaging 
and delivery efficiency while addressing ethical and privacy challenges in 
urban planning.
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

# Generate synthetic data for urban delivery route optimization
data = {
    'delivery_distance': np.random.rand(1000) * 50,  # Distance in kilometers
    'delivery_time': np.random.rand(1000) * 120,     # Time in minutes
    'traffic_conditions': np.random.randint(1, 5, 1000),  # Traffic condition rating 1-4
    'num_stops': np.random.randint(1, 10, 1000),     # Number of stops
    'historical_cost': np.random.rand(1000) * 100,   # Historical delivery cost
    'real_time_cost': np.random.rand(1000) * 100     # Real-time delivery cost
}
df = pd.DataFrame(data)

# Split the data into features and target
X = df.drop(['real_time_cost'], axis=1)
y = df['real_time_cost']

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
