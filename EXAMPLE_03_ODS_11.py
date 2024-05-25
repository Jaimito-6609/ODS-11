# -*- coding: utf-8 -*-
"""
Created on Fri May 24 18:25:49 2024

@author: Jaime
"""
#==============================================================================
# EXAMPLE 03 ODS 11
#==============================================================================
"""
Duan et. al (2022) apply artificial intelligence, including Generative 
Adversarial Network (GAN) and Natural Language Processing (NLP), in the 
sustainable reuse of urban ruins. They develop a planning strategy to 
transform areas such as abandoned industrial sites and war zones. Using 
studies in Guangzhou and combining AI with big data, they create an innovative 
and globally applicable method to reconstruct and optimize these spaces, 
improving the urban image and the well-being of residents.

Duan, J., Zhang, L., & Li, H. (2022). AI-driven Sustainable Urban Ruins Reuse: 
A Case Study in Guangzhou. Urban Planning and Development, 148(4), 03021013. 
https://doi.org/10.1061/(ASCE)UP.1943-5444.0000725.

This Python code integrates Generative Adversarial Networks (GAN) with a 
genetic algorithm to optimize the sustainable reuse of urban ruins. The GAN is 
used to generate realistic site descriptions and sustainability scores, 
leveraging the power of AI in transforming abandoned industrial sites and war 
zones into sustainable urban spaces. The genetic algorithm, implemented using 
the DEAP library, optimizes the hyperparameters of the GAN (epochs and batch 
size) to minimize the root mean square error (RMSE) on the sustainability score 
predictions. The synthetic dataset includes various features relevant to urban 
planning, such as site descriptions and sustainability scores. The evolutionary 
algorithm evolves a population of individuals, each representing a set of 
hyperparameters, to find the best combination that optimizes the GAN's 
performance.
"""

import numpy as np
import pandas as pd
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from deap import base, creator, tools, algorithms
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, Reshape, Flatten, Conv2D, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense

# Generate synthetic data for urban ruins reuse planning
data = {
    'site_description': [
        'abandoned industrial site with potential for green space',
        'war zone requiring rehabilitation and community space',
        'old factory building suitable for conversion to residential area',
        # ... (add more descriptions)
    ] * 100,
    'sustainability_score': np.random.rand(300) * 100
}
df = pd.DataFrame(data)

# Vectorize the text descriptions using NLP techniques
vectorizer = CountVectorizer()
X_text = vectorizer.fit_transform(df['site_description']).toarray()

# Split the data into features and target
X = X_text
y = df['sustainability_score']

# Normalize the target
y = y / 100.0

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the GAN architecture
def build_generator(latent_dim, output_shape):
    model = Sequential()
    model.add(Dense(256, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(np.prod(output_shape), activation='tanh'))
    model.add(Reshape(output_shape))
    return model

def build_discriminator(input_shape):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    return model

# GAN parameters
latent_dim = 100
output_shape = (X_train.shape[1],)

# Build and compile the discriminator
discriminator = build_discriminator(output_shape)
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])

# Build the generator
generator = build_generator(latent_dim, output_shape)

# Define the combined GAN model
z = Input(shape=(latent_dim,))
img = generator(z)
discriminator.trainable = False
valid = discriminator(img)
combined = Model(z, valid)
combined.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

# Define the evaluation function for the genetic algorithm
def evaluate(individual):
    epochs, batch_size = individual

    # Train the GAN model
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(int(epochs)):
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        gen_imgs = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(imgs, valid)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        g_loss = combined.train_on_batch(noise, valid)

    # Evaluate the generator's performance on the test set
    noise = np.random.normal(0, 1, (X_test.shape[0], latent_dim))
    gen_imgs = generator.predict(noise)
    gen_imgs = np.argmax(gen_imgs, axis=1)
    rmse = np.sqrt(mean_squared_error(y_test, gen_imgs))
    return (rmse,)

# Set up genetic algorithm tools
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_int", random.randint, 1, 100)
toolbox.register("individual", tools.initCycle, creator.Individual, 
                 (toolbox.attr_int, toolbox.attr_int), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=1, up=100, indpb=0.2)
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
