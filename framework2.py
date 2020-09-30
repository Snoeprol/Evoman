import sys, os
sys.path.insert(0, 'evoman')
from environment import Environment
import numpy as np
import scipy.stats as sp
import random
from demo_controller import player_controller
#import matplotlib.pyplot as plt
from covariance import mutate 
import math
import pandas as pd
import time
import os
cwd = os.getcwd()
sys.argv = [1, '1', '1']
#os.putenv('SDL_VIDEODRIVER', 'fbcon')
#os.environ["SDL_VIDEODRIVER"] = "dummy"

min_weight = -1
max_weight = 1

def time_it(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('Function time ' + method.__name__ + ': ' + str(round((te - ts) * 1000,7)) + 'ms')
        return result
    return timed

class Individual:
    def __init__(self, weights, stddevs):
        self.weights = weights
        self.stddevs = stddevs 
        self.fitness = 0

    def evaluate(self, env):
        self.fitness = simulation(env, self.weights)

    def evaluate_multi(self, bosses):
        fitness = 0
        for boss_number in bosses:
            env = Environment(experiment_name="test123",
                        playermode="ai",
                        player_controller=player_controller(hidden),
                        enemies = [boss_number],
                        speed="fastest",
                        enemymode="static",
                        level=2)
            fitness += simulation(env, self.weights)
        self.multi_fitness = fitness

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
            if self.stddevs[i] > 10 * stddev_lim:
                self.stddevs[i] == 10 * stddev_lim
    
    def log(self):
        with open("best_multi.txt",'w') as f:
            f.write('Fitness, {}, weighths, {}'.format(self.multi_fitness,self.weights))

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

def select_ranking(fitness_list, s = 1.5):
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
    
def alpha_adapt(number_of_zero_alpha_generations, generation):
    """
    number of zero alpha generations is the amount of generations at the end of the runs with alpha = 0
    """
    global ALPHA, generations, rate_of_change
    if generation == 0:
        rate_of_change = ALPHA / (generations - number_of_zero_alpha_generations) #lineral rate of change to zero for alpha
    
    ALPHA -= rate_of_change

   
def save_pop(pop):
    list_of_values = []
    #create dataframe to be save as csv
    amount_of_weights = len(pop[0].weights) #get length of the df
    header = [] #create header csv
    for i in range(amount_of_weights):
        header.append(f'Weight {i}')
    for n in range(amount_of_weights):
        header.append(f'STD DEV {n}')
    header.append('Fitness')
          
    
    #loop over individuals
    for indi in pop:
        
        indi_attributes = list(np.append(indi.weights, indi.stddevs))
        indi_attributes.append(indi.fitness)
        list_of_values.append(indi_attributes)
    
    df_to_csv = pd.DataFrame(list_of_values, columns = header)
    
    df_to_csv.to_csv(f'OutputData/Enemy {bosses}, Generation {generation}, Max Fitness {round(max(fitness_list),2)}, Average {round(np.mean(fitness_list),2)}, Hidden nodes {hidden}, {sys.argv[2]}, Unique Runcode {unique_runcode}.csv')

#if __name__ ==  '__main__':
if True == True:
    global tau, tau_2, beta, stddev_lim, ALPHA, bosses
    hidden = 10
    population_size = 2
    generations = 50
    ALPHA = 0.5
    adapt_alpha = False
    tau = 1/np.sqrt(2 * population_size)
    tau_2 = 1/np.sqrt(np.sqrt(population_size))
    stddev_lim = 0.05
    bosses = [1,2,3,4]
    tournament = 1
    max_fitness = -1000

    for q in range(1):
        unique_runcode = random.random()

        n_vars = (20+1)*hidden + (hidden + 1)*5        
        max_fitness_per_gen = []
        average = []
        pop = initiate_population(population_size,n_vars, -1, 1)

        stats_per_gen = []
        for generation in range(generations):
        
            for individual in pop:
                individual.evaluate_multi(bosses)
                with open("best_multi.txt",'r') as f:
                    max_fitness =  float(f.readline().split(',')[1])
                    
                if individual.multi_fitness > max_fitness:
                    max_fitness = individual.multi_fitness
                    individual.log()

            fitness_list = np.array([individual.fitness for individual in pop])

            new_pop = []
            for _ in range(population_size // 2):
                if tournament == '1':
                    parent_index_1 = select_tournament(fitness_list, 2)
                    parent_index_2 = select_tournament(fitness_list, 2)
                else:
                    parent_index_1 = select_ranking(fitness_list)
                    parent_index_2 = select_ranking(fitness_list)

                ind1, ind2 = blend_crossover(pop[parent_index_1], pop[parent_index_2])
                ind1.check_and_alter_boundaries()
                ind2.check_and_alter_boundaries()
                
                new_pop.extend([ind1, ind2])

            for individual in new_pop:
                if np.random.rand() > 0.7:
                    individual.mutate_self()
                    individual.check_and_alter_boundaries()
            pop = new_pop
            
            if adapt_alpha:
                alpha_adapt(5, generation)
                    


            print('New generation of degenerates eradicated.')
            max_fitness_per_gen.append(max(fitness_list))
            average.append(np.mean(fitness_list))
            stats_per_gen.append([np.mean(fitness_list), np.max(fitness_list), np.min(fitness_list)])

        for i in range(len(stats_per_gen)):
            print("GEN {}, max = {:.2f}, min = {:.2f}, mean = {:.2f}".format(i, stats_per_gen[i][1], stats_per_gen[i][2], stats_per_gen[i][0]))
            
        print(max_fitness_per_gen)
