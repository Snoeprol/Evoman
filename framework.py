import sys, os
sys.path.insert(0, 'evoman')
from environment import Environment
import numpy as np
import scipy.stats as sp
import random
from demo_controller import player_controller
import matplotlib.pyplot as plt
from covariance import mutate 
import math
import pandas as pd

os.putenv('SDL_VIDEODRIVER', 'fbcon')
os.environ["SDL_VIDEODRIVER"] = "dummy"

min_weight = -1
max_weight = 1

class Individual:
    def __init__(self, weights, stddevs):
        self.weights = weights
        self.stddevs = stddevs 
        self.fitness = 0

    def evaluate(self, env):
        self.fitness = simulation(env, self.weights)

    def mutate_self(self):
        mutate(self, tau, tau_2)

    def check_and_alter_boundaries(self):
            
        for i in range(len(self.weights)):
            if self.weights[i] < -1:
                self.weights[i] = -1
            if self.weights[i] > 1:
                self.weights[i] = 1
        
        for i in range(len(self.stddevs)):
            if self.stddevs[i] < stddev_lim:
                self.stddevs[i] == stddev_lim


def initiate_population(size, variables, min_weight, max_weight):
    ''' Initiate a population of individuals with variables amount of parameters unfiformly 
    chosen between min_weight and max_weight'''
    population = []

    #stddevs = [2/np.sqrt(12)] * variables
    stddevs = [stddev_lim] * variables 
    for _ in range(size):
        weights = np.random.rand(variables) * (max_weight - min_weight) +  min_weight
        weights_1 = np.random.uniform(min_weight, max_weight, variables)
  
        population.append(Individual(weights, np.array(stddevs)))

    return population



def blend_crossover(ind1, ind2):
    """
    Blend two genomes to two offsprings
    Code taken from:
    https://github.com/DEAP/deap/blob/master/deap/tools/crossover.py
    """
    ind1_list = [ind1.weights, ind1.stddevs]
    ind2_list = [ind2.weights, ind2.stddevs]
    
    for position_genome, (mixed_tuple_1, mixed_tuple_2) in enumerate(zip(ind1_list, ind2_list)):
        #add random factor for exploration
        beta = (1. + 2. * ALPHA) * random.random() - ALPHA
        #add genomes to kids
        ind1_list[position_genome] = (1. - beta) * mixed_tuple_1 + beta * mixed_tuple_2
        ind2_list[position_genome] = beta * mixed_tuple_1 + (1 - beta) * mixed_tuple_2

    return_ind1 = Individual(ind1_list[0], ind1_list[1])
    return_ind2 = Individual(ind2_list[0], ind2_list[1])

    return return_ind1, return_ind2

def select_best(fitness_list, n):
    '''Returns indices of n best individuals'''
    return (-fitness_list).argsort()[:n]

    
def select_tournament(fitness_list, tour_size):
    
    '''Takes fitness of the population, chooses 
    tour_size individuals at random
    and returns the index of the best individual'''
    #randomly select indexes of individuals
    chosen_indexes = [np.random.randint(0, len(fitness_list)) for iter in range(tour_size)]
    # gets fitness of chosen individuals
    chosen_population = fitness_list[chosen_indexes]
    # returns an index of the best individual from chosen individuals
    max_individual = np.argmax(chosen_population)
    # returns an index of the selected individuals from the whole population
    # print(max_individual, chosen_indexes[max_individual], "tour")
    return chosen_indexes[max_individual]


def select_ranking(fitness_list, s):
    ''' Ranks individuals according to its fitness and assigns a probability to
    an individual index according to its place in the ranking
    
    s controls the selection aggressiveness towards the best individual: 1<s<2
    '''
    # returns ranking of individuals (starting from 0)
    ranking = sp.rankdata(fitness_list) - 1
    # calculates probabilities for individuals based on eq. in page 82
    probabilities = [(2-s)/len(ranking) + (2*indiv*(s-1))/(len(ranking)*(len(ranking) - 1)) for indiv in ranking]
    # returns an index of randomly chosen individual based on probabilities provided
    return np.random.choice(len(fitness_list), p=probabilities) 
    

def calculate_fitness(fitness_list):
    '''Calculated the total fitness of a population by summing up their
    values'''
    total_fitness = 0
    for i in fitness_list:
        total_fitness += i
    
    return total_fitness


def simulation(env,x):
    f,p,e,t = env.play(x)
    return f

def cross_over_test(pop):
    """
    Return the amounts of individuals with values outside the boundaries for the crossover
    """
    counter = 0
    for i in range(len(pop) - 1):
        a,b = blend_crossover(pop[i], pop[i+1])
        if (max(a.weights) > 1) or min((a.weights) < -1):
            counter += 1
        
        if (max(b.weights) > 1) or min((b.weights) < -1):
            counter += 1
            
        print(a.weights)
        print('\n')
        print(b.weights)
    
    print(counter)
   
def save_pop(pop):
    list_of_values = []
    #create dataframe to be save as csv
    amount_of_weights = len(pop[0].weights) #get length of the df
    header = [] #create header csv
    for i in range(amount_of_weights):
        header.append(f'Weight {i}')
    for n in range(amount_of_weights):
        header.append(f'STD DEV {n}')
          
    
    #loop over individuals
    for indi in pop:
        
        indi_attributes = list(np.append(indi.weights, indi.stddevs))
        list_of_values.append(indi_attributes)
    
    df_to_csv = pd.DataFrame(list_of_values, columns = header)
    
    df_to_csv.to_csv(f'OutputData/Generation {generation}, Max Fitness {max(fitness_list)}, Unique Runcode {unique_runcode}.csv')

if __name__ ==  '__main__':
    global tau, tau_2, beta, stddev_lim, ALPHA
    hidden = 1
    population_size = 5
    generations = 20
    ALPHA = 0.5
    tau = 1/np.sqrt(2 * population_size)
    tau_2 = 1/np.sqrt(np.sqrt(population_size))
    stddev_lim = 0.05
    #beta = 5/ 360 * 2 * np.pi
    unique_runcode = random.random()

    env = Environment(experiment_name="test123",
                  playermode="ai",
                  player_controller=player_controller(hidden),
                  enemies = [2],
                  speed="fastest",
                  enemymode="static",
                  level=2)

    n_vars = (env.get_num_sensors()+1)*hidden + (hidden + 1)*5        
    max_fitness_per_gen = []
    average = []
    pop = initiate_population(population_size,n_vars, -1, 1)

    stats_per_gen = []
    for generation in range(generations):

        for individual in pop:
            individual.evaluate(env)

        fitness_list = np.array([individual.fitness for individual in pop])

        new_pop = []
        for _ in range(population_size // 2):
            parent_index_1 = select_tournament(fitness_list, 2)
            parent_index_2 = select_tournament(fitness_list, 2)
            # print(parent_index_1, parent_index_2, pop[parent_index_1].fitness, pop[parent_index_2].fitness, "test")
            ind1, ind2 = blend_crossover(pop[parent_index_1], pop[parent_index_2])
            ind1.check_and_alter_boundaries()
            ind2.check_and_alter_boundaries()
            
            new_pop.extend([ind1, ind2])

        for individual in new_pop:

            individual.mutate_self()
            individual.check_and_alter_boundaries()
        save_pop(pop)
        pop = new_pop


        print('New generation of degenerates eradicated.')
        max_fitness_per_gen.append(max(fitness_list))
        average.append(np.mean(fitness_list))
        stats_per_gen.append([np.mean(fitness_list), np.max(fitness_list), np.min(fitness_list)])

    for i in range(len(stats_per_gen)):
        print("GEN {}, max = {:.2f}, min = {:.2f}, mean = {:.2f}".format(i, stats_per_gen[i][1], stats_per_gen[i][2], stats_per_gen[i][0]))
    print(max_fitness_per_gen)
    plt.plot(average)
    plt.show()
