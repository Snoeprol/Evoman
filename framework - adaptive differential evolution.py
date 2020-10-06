import sys, os
sys.path.insert(0, 'evoman')
from environment import Environment
import numpy as np
import scipy.stats as sp
import random
from demo_controller import player_controller
from numpy.random import multivariate_normal
import pandas as pd
import ast # read a data frame with lists
import matplotlib.pyplot as plt 
import math
import copy

os.putenv('SDL_VIDEODRIVER', 'fbcon')
os.environ["SDL_VIDEODRIVER"] = "dummy"

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
    def __init__(self, weights, F):
        self.weights = weights
        self.F = F
        self.best = weights
        self.multi_fitness = -100

    def evaluate(self, env):
        self.fitness = simulation(env, self.weights)
        
    def evaluate_multi(self, bosses):
        # Changed to gain
        total_fitness = 0
        for boss_number in bosses:
            env = Environment(experiment_name="test123",
                        playermode="ai",
                        player_controller=player_controller(hidden),
                        enemies = [boss_number],
                        speed="fastest",
                        enemymode="static",
                        level=2)
            values = simulation_gain(env, self.weights)
            total_fitness += values[0] - values[1]

        self.multi_fitness = total_fitness
        
    def check_and_alter_boundaries(self):
            
        for i in range(len(self.weights)):
            if self.weights[i] < -1:
                self.weights[i] = -1
            if self.weights[i] > 1:
                self.weights[i] = 1     
    
    def log(self):
        with open("best_multi.txt",'w') as f:
            f.write('Fitness, {}, weighths, {}'.format(self.multi_fitness,self.weights))


def initiate_population(size, variables, min_weight, max_weight):
    ''' Initiate a population of individuals with variables amount of parameters unfiformly 
    chosen between min_weight and max_weight'''
    population = []

    for _ in range(size):
        weights = np.array(np.random.rand(variables) * (max_weight - min_weight) +  min_weight)
        F = np.random.normal(0.5, 0.15)
        population.append(Individual(weights, F))

    return population


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

def simulation_gain(env,x):
    f,p,e,t = env.play(x)
    return p, e
   
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

def save_pop2(pop):
     
    weights = []
    multi_fitness = []
    for individual in pop:     
         weights.append(individual.weights)
         multi_fitness.append(individual.multi_fitness)
    
    pandas_dict = {"multi_fitness": multi_fitness,
                   "weights": weights}
    
    df_to_csv = pd.DataFrame(pandas_dict)
    
    df_to_csv.to_csv(f'D, OutputData/Enemy {bosses}, Generation {generation}, Max Fitness {round(max(multi_fitness),2)}, Average {round(np.mean(multi_fitness),2)}, Hidden nodes {hidden}, {sys.argv[2]}, Unique Runcode {unique_runcode}.csv')

def read_data(file_path):
    def from_np_array(array_string):
        array_string = ','.join(array_string.replace('[ ', '[').split())
        return np.array(ast.literal_eval(array_string))
    return pd.read_csv(file_path, converters={'weights': from_np_array})
    
def adapt_F(pop, target_indx):
    '''takes pop and target individual index and returns "mutated" F,
    also truncates F if it is outside (0, 1]
    '''
    valid_choices = [j for j in range(len(pop)) if j != target_indx]
    selections = random.sample(valid_choices, 3)
    f1, f2, f3 = pop[selections[0]].F, pop[selections[1]].F, pop[selections[2]].F
    new_F = f1 + np.random.normal(0, 0.5) * (f2 - f3)
    # adjust F so it is in (0, 1]
    new_F = abs(new_F) - abs(math.trunc(new_F))
    return new_F

def mutate_diff(pop, target_indx, F, mode = "DE/rand/1"):
    '''takes population, index of the target individual and mode
    returns mutant vector
    '''
    if mode == "DE/rand/1" or mode == 0:
        
        valid_choices = [j for j in range(len(pop)) if j != target_indx]
        # select target a, rand_indv1 b, rand_ind2 c
        selections = random.sample(valid_choices, 3)
        a, b, c = pop[selections[0]].weights, pop[selections[1]].weights, pop[selections[2]].weights
        #create a new mutation
        new_ind = Individual(a + F * (b - c), F)
        new_ind.check_and_alter_boundaries() 
        
    if mode == "DE/rand-to-best/2" or mode == 1:
        
        fitness_list = np.array([individual.multi_fitness for individual in pop])
        best_solution = np.argmax(fitness_list)
        valid_choices = [j for j in range(len(pop)) if (j != target_indx and j != best_solution)]
        selections = random.sample(valid_choices, 4)
        a, b, c, d = pop[selections[0]].weights, pop[selections[1]].weights, pop[selections[2]].weights, pop[selections[3]].weights
        new_ind = Individual(pop[target_indx].weights + F * (pop[best_solution].weights - pop[target_indx].weights) + F * (a - b) + F * (c - d), F)
        new_ind.check_and_alter_boundaries() 
    if mode == "DE/rand/2" or mode == 2:
        
        valid_choices = [j for j in range(len(pop)) if j != target_indx]
        # select target a, rand_indv1 b, rand_ind2 c
        selections = random.sample(valid_choices, 5)
        a, b, c, d, e = pop[selections[0]].weights, pop[selections[1]].weights, pop[selections[2]].weights, pop[selections[3]].weights, pop[selections[4]].weights
        #create a new mutation
        new_ind = Individual(a + F * (b - c) + F * (d - e), F)
        new_ind.check_and_alter_boundaries() 
        
    if mode == "DE/current-to-rand" or mode == 3:
        
        K = 0.4
        valid_choices = [j for j in range(len(pop)) if j != target_indx]
        # select target a, rand_indv1 b, rand_ind2 c
        selections = random.sample(valid_choices, 3)
        a, b, c = pop[selections[0]].weights, pop[selections[1]].weights, pop[selections[2]].weights
        new_ind = Individual(pop[target_indx].weights + K * (a - pop[target_indx].weights) + F * (b - c), F)
        new_ind.check_and_alter_boundaries() 
        
    return new_ind

def uni_crossover_fixed(ind_target, ind_mutant):
    '''perform uniform crossover with one fixed allel
    (to ensure offspring is always different from the target vector),
    '''
    ind_size = len(ind_target.weights)
    ind_target_copy = copy.deepcopy(ind_target)
    ind_mutant_copy = copy.deepcopy(ind_mutant)
    # define fixed allel and swap it into target individual from mutant individual
    fixed_allel = random.choice(range(ind_size))
    ind_target_copy.weights[fixed_allel] = ind_mutant_copy.weights[fixed_allel]
    crossover_prob = np.random.normal(0.5, 0.15)
    for i in range(len(ind_target.weights)):
        if random.random() < crossover_prob:
            ind_target_copy.weights[i] = ind_mutant_copy.weights[i]
    # update offspring F to new_F
    ind_target_copy.F = ind_mutant_copy.F
    return ind_target_copy

if __name__ ==  '__main__':

    hidden = 10
    population_size = 100
    generations = 50
    bosses = [2,5,6]

    n_vars = (20+1)*hidden + (hidden + 1)*5 

    upper_bound = 1
    lower_bound = -1
    runs = 1


    for _ in range(runs):
        unique_runcode = random.random()
        
        #initialise population
        pop = initiate_population(population_size, n_vars, lower_bound, upper_bound)
        
        # evaluate random population
        for individual in pop:
            individual.evaluate_multi(bosses)


        for generation in range(generations):
            # save population
            save_pop2(pop)
            # get average fitness
            print("Generation: " + str(generation))
            new_pop = []
            for target_indx in range(len(pop)):
                # create a mutant
                new_F = adapt_F(pop, target_indx)
                mutant = mutate_diff(pop, target_indx, F = new_F)
                # perform crossover to create trial individual
                trial = uni_crossover_fixed(pop[target_indx], mutant)
                #evaluate the new trial individual
                trial.evaluate_multi(bosses)
                # choose who survives, trial or parent
                if trial.multi_fitness > pop[target_indx].multi_fitness:
                    new_pop.append(trial)
                else:
                    new_pop.append(pop[target_indx])
                # assign new_F to the new individual in the new_pop
            pop = new_pop
 
