import numpy as np
from numpy.random import multivariate_normal

tau_2 = 0.01
tau = 0.01
beta = 0.001

def mutate(individual):
    '''Assumes individual of the form   individual.weights = weights/ biases
                                        individual.stddevs = standard devs'''
    individual.stddevs *= np.exp(tau_2 * np.random.normal(0,1, len(individual.stddevs)) + tau * np.random.normal(0,1, len(individual.stddevs)))
    change = multivariate_normal(mean = np.zeros(len(individual.weights)), cov = np.diag(individual.stddevs))
    individual.weights += change
    return individual
