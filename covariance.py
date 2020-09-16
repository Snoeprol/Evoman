import numpy as np
from scipy.stats import multivariate_normal

tau_2 = 0.01
tau = 0.01
beta = 0.001

class Individual:
    def __init__(self, weights, stddevs, alphas):
        self.weights = weights
        self.stddevs = stddevs 
        self.alphas = alphas 
        self.fitness = 0

    def evaluate(self, env):
        self.fitness = simulation(env, self.weights)

    def mutate_self(self):
        mutate(self)

def mutate(individual):
    '''Assumes individual of the form   individual[0] = weights/ biases
                                        individual[1] = standard devs
                                        individual[2] = alphas, in n (n - 1) /2'''
    # Assume tau, tau_2 , beta are defined
    individual.stddevs *= np.exp(tau_2 * np.random.normal(0,1, len(individual.stddevs)) + tau * np.random.normal(0,1, len(individual.stddevs)))
    '''
    for stdev in individual[1]:
        x = tau_2 * np.random.normal(0,1) + tau * np.random.normal(0,1)
        stdev *= np.exp(x)
    '''
    individual.alphas += beta * np.random.normal(0,1, len(individual.alphas))
    covariance = generate_covariance_matrix(individual.stddevs, individual.alphas)
    change = multivariate_normal(np.zeros(len(individual.weights)), cov = covariance).rvs()
    individual.weights += change
    return individual

def generate_covariance_matrix(st_devs, angles):
    columns = rows = len(st_devs)
    covariance_matrix = np.zeros((rows, columns))
    for index, row in enumerate(covariance_matrix):
        for index_2, number in enumerate(row):
            if index == index_2:
                covariance_matrix[index][index_2] = st_devs[index] ** 2
            else:
                # print(angles.size)
                # print(int(index * (index_2 -1) / 2))
                # print(angles[int(index * (index_2 -1) / 2)])
                covariance_matrix[index][index_2] = covariance_matrix[index_2][index] = (st_devs[index] ** 2 - st_devs[index_2] ** 2) /2 * np.tan(angles[int(index * (index_2 -1) / 2)])
    print(covariance_matrix)
    return covariance_matrix

# x = Individual([-1,1,-1,1],[-1,-1,-1,-1],[-1,1,1,-1,1,1])
# print(mutate(x))