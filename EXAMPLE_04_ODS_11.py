# -*- coding: utf-8 -*-
"""
Created on Wed May 15 19:34:43 2024

@author: Jaime
"""
#==============================================================================
# EXAMPLE 04 ODS 11
#==============================================================================

"""
Liu et. al (2023) analyze how artificial intelligence (AI), through data 
analysis and route optimization, revolutionizes waste management in urban 
environments, supporting cleaner cities and a lower carbon footprint. Within 
the framework of sustainable urban development, they use geographic information 
systems (GIS) and genetic algorithms to create street designs that balance the 
needs of pedestrians and vehicular traffic, promoting healthy and sustainable 
environments.

Liu, Y., Qin, S., Li, J., & Jin, T. (2023). Artificial Intelligence and Street 
Space Optimization in Green Cities: New Evidence from China. Sustainability, 
15(23), 16367. https://doi.org/10.3390/su152316367.

This Python code integrates geographic information systems (GIS) with a 
genetic algorithm to optimize waste management routes in urban environments, 
supporting sustainable urban development. The code simulates waste location 
data and uses the DEAP library to implement a genetic algorithm that minimizes 
the total travel distance for waste collection. The genetic algorithm optimizes 
the order of visits to waste locations, considering geographic distances. The 
optimal route is visualized using the Folium library and saved as an HTML file. 
This approach balances the needs of pedestrians and vehicular traffic, 
promoting healthy and sustainable environments.

"""

import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from deap import base, creator, tools, algorithms
import random

# Simulate waste location data
data = {
    'id': range(10),
    'lat': [37.77 + random.uniform(-0.01, 0.01) for _ in range(10)],
    'lon': [-122.42 + random.uniform(-0.01, 0.01) for _ in range(10)]
}
waste_df = pd.DataFrame(data)
gdf = gpd.GeoDataFrame(waste_df, geometry=gpd.points_from_xy(waste_df.lon, waste_df.lat))

# Define the evaluation function for the genetic algorithm
def evaluate(individual):
    # Assume individual is a list of indices representing the order of visits
    distance = 0
    for i in range(1, len(individual)):
        point1 = gdf.iloc[individual[i-1]].geometry
        point2 = gdf.iloc[individual[i]].geometry
        distance += point1.distance(point2)
    return (distance,)

# Set up genetic algorithm tools
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("indices", random.sample, range(len(gdf)), len(gdf))
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

# Run the genetic algorithm
population = toolbox.population(n=50)
hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("min", np.min)
stats.register("max", np.max)

result = algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=40, stats=stats, halloffame=hof, verbose=True)

# Visualize the optimal route
best_route = hof[0]
map = folium.Map(location=[37.77, -122.42], zoom_start=13)
for idx in best_route:
    folium.Marker(location=[gdf.iloc[idx].lat, gdf.iloc[idx].lon]).add_to(map)
folium.PolyLine([(gdf.iloc[idx].lat, gdf.iloc[idx].lon) for idx in best_route], color="red").add_to(map)
map.save("optimized_route.html")

print("The optimal route has been saved as 'optimized_route.html'.")
