import numpy as np
from numpy.random import multivariate_normal

def mutate(individual):
    '''Assumes individual of the form   individual.weights = weights/ biases
                                        individual.stddevs = standard devs'''

    tau = 0.0001
    tau_2 = 0.00001
    
    individual.stddevs *= np.exp(tau_2 * np.random.normal(0,1, len(individual.stddevs)) + tau * np.random.normal(0,1, len(individual.stddevs)))
    change = multivariate_normal(mean = np.zeros(len(individual.weights)), cov = np.diag(individual.stddevs))
    individual.weights += change
    return individual
