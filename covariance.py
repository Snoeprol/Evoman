import numpy as np
from scipy.stats import multivariate_normal

def mutate(individual):
    '''Assumes individual of the form   individual[0] = weights/ biases
                                        individual[1] = standard devs
                                        individual[2] = alphas, in n (n - 1) /2'''
    # Assume tau, tau_2 , beta are defined
    individual[1] *= np.exp(tau_2 * np.random.normal(0,1, len(individual[1])) + tau * np.random.normal(0,1, len(individual[1])))
    '''
    for stdev in individual[1]:
        x = tau_2 * np.random.normal(0,1) + tau * np.random.normal(0,1)
        stdev *= np.exp(x)
    '''
    individual[2] += beta * np.random.normal(0,1, len(individual[2]))
    covariance = generate_covariance_matrix(individual[1], individual[2])
    change = multivariate_normal(np.array(len(individual[0])), cov = x).rvs()
    individual[0] += change

def generate_covariance_matrix(st_devs, angles):
    columns = rows = len(st_devs)
    covariance_matrix = np.zeros((rows, columns))
    for index, row in enumerate(covariance_matrix):
        for index_2, number in enumerate(row):
            if index == index_2:
                covariance_matrix[index][index_2] = st_devs[index] ** 2
            else:
                covariance_matrix[index][index_2] = covariance_matrix[index_2][index] = (st_devs[index] ** 2 - st_devs[index_2] ** 2) /2 * np.tan(angles[int(index * (index_2 -1) / 2)])

    return covariance_matrix


print(x)
mutate()