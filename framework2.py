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
cwd = os.getcwd()
sys.argv = [1, '1', '1']
os.putenv('SDL_VIDEODRIVER', 'fbcon')
os.environ["SDL_VIDEODRIVER"] = "dummy"

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
    def __init__(self, weights, velocities):
        self.weights = weights
        self.velocities = velocities
        self.best = weights
        self.best_fitness = -100 
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
        if total_fitness > self.best_fitness:
        	self.best_fitness = total_fitness
        	self.best = self.weights

    def check_and_alter_boundaries(self):
            
        for i in range(len(self.weights)):
            if self.weights[i] < -1:
                self.weights[i] = -1
            if self.weights[i] > 1:
                self.weights[i] = 1
    
    def log(self):
        with open("best_multi.txt",'w') as f:
            f.write('Fitness, {}, weighths, {}'.format(self.multi_fitness,self.weights))


def initiate_population(size, variables, min_weight, max_weight, velocity):
    ''' Initiate a population of individuals with variables amount of parameters unfiformly 
    chosen between min_weight and max_weight'''
    population = []

    velocities = [velocity] * variables 
    for _ in range(size):
        weights = np.random.rand(variables) * (max_weight - min_weight) +  min_weight
        population.append(Individual(weights, np.array(velocity)))

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
    
    df_to_csv.to_csv(f'OutputData/Enemy {bosses}, Generation {generation}, Max Fitness {round(max(multi_fitness),2)}, Average {round(np.mean(multi_fitness),2)}, Hidden nodes {hidden}, {sys.argv[2]}, Unique Runcode {unique_runcode}.csv')

def read_data(file_path):
    def from_np_array(array_string):
        array_string = ','.join(array_string.replace('[ ', '[').split())
        return np.array(ast.literal_eval(array_string))
    return pd.read_csv(file_path, converters={'weights': from_np_array})
    
def mutate_swarm(individual, global_best):

    # Generate random matrices
    U_1 = []
    U_2 = []
    U_1_sum = U_2_sum = 0

    for i in range(len(individual.weights)):
        U_i_1 = np.random.random()
        U_i_2 = np.random.random()
        U_1.append(U_i_1)
        U_2.append(U_i_2)
        U_1_sum += U_i_1
        U_2_sum += U_i_2
    
    U_1 = np.diagflat(np.array(U_1)/U_1_sum)
    U_2 = np.diagflat(np.array(U_2)/U_2_sum)
    
    # Define weights
    w1 = 0.4
    w2 = 0.3
    w3 = 0.3
    
    vec_1 = individual.best - individual.weights
    vec_2 = global_best - individual.weights
    # Add vectors
    individual.velocities = w1 * individual.velocities + w2 * U_1 * vec_1 + w3 * U_2 * vec_2 
    individual.weights = individual.weights + individual.velocities


if __name__ ==  '__main__':
    global tau, tau_2, beta, stddev_lim, ALPHA, bosses
    hidden = 10
    population_size = 2
    generations = 4

    n_vars = (20+1)*hidden + (hidden + 1)*5 

    bosses = [1,2,3,4]
    tournament = 1
    max_fitness = -1000
    upper_bound = 1
    lower_bound = -1
    velocity = 1.2
    for q in range(1):
        unique_runcode = random.random()

 
        max_fitness_per_gen = []
        average = []
        pop = initiate_population(population_size, n_vars, lower_bound, upper_bound, velocity)

        stats_per_gen = []
        for generation in range(generations):
        
            for individual in pop:
                individual.evaluate_multi(bosses)
                with open("best_multi.txt",'r') as f:
                    max_fitness =  float(f.readline().split(',')[1])

                    
                if individual.multi_fitness > max_fitness:
                    max_fitness = individual.multi_fitness
                    individual.log()

            fitness_list = np.array([individual.multi_fitness for individual in pop])

            # new_pop = []

            # pop = new_pop

                    


            print('New generation of degenerates eradicated.')
            max_fitness_per_gen.append(max(fitness_list))
            average.append(np.mean(fitness_list))
            stats_per_gen.append([np.mean(fitness_list), np.max(fitness_list), np.min(fitness_list)])

        for i in range(len(stats_per_gen)):
            print("GEN {}, max = {:.2f}, min = {:.2f}, mean = {:.2f}".format(i, stats_per_gen[i][1], stats_per_gen[i][2], stats_per_gen[i][0]))
            
        print(max_fitness_per_gen)
individual = Individual(np.random.rand(100), npp.random.rand(100))
global_best = np.random.rand(100)
mutate_swarm(individual, global_best)
