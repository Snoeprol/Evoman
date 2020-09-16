import sys
sys.path.insert(0, 'evoman')
from environment import Environment
import numpy as np
import scipy.stats as sp
import random
from demo_controller import player_controller
import matplotlib.pyplot as plt
from covariance import mutate 
import math
# os.putenv('SDL_VIDEODRIVER', 'fbcon')
# os.environ["SDL_VIDEODRIVER"] = "dummy"

min_weight = -1
max_weight = 1

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

    def check_and_alter_boundaries(self):
        #check if all the weight values are between -1, 1
        if (min(self.weights) < -1) or (max(self.weights) > 1):
            #not the case, change the values to max allowed value
            for i in self.weights:
                if i < -1:
                    i = -1
                if i > 1:
                    i = 1
        
        
        if (min(self.alphas) < - math.pi) or (max(self.alphas) > math.pi):
            for i in self.alphas:
                if i < -math.pi:
                    i = -math.pi
                if i > math.pi:
                    i = math.pi

def initiate_population(size, variables, min_weight, max_weight):
    ''' Initiate a population of individuals with variables amount of parameters unfiformly 
    chosen between min_weight and max_weight'''
    population = []

    stddevs = [2/np.sqrt(12)] * variables

    for _ in range(size):
        weights = np.random.rand(variables) * (max_weight - min_weight) +  min_weight
        alphas_indiv = []
        for _ in range(int(variables * (variables - 1) / 2)):
            alphas_indiv.append(np.random.uniform(-np.pi, np.pi))
        population.append(Individual(weights, np.array(stddevs), np.array(alphas_indiv)))

    return population



def blend_crossover(ind1, ind2):
    """
    Blend two genomes to two offsprings
    Code taken from:
    https://github.com/DEAP/deap/blob/master/deap/tools/crossover.py
    """

    ind1_list = [ind1.weights, ind1.stddevs, ind1.alphas]
    ind2_list = [ind2.weights, ind2.stddevs, ind2.alphas]
    
    for position_genome, (mixed_tuple_1, mixed_tuple_2) in enumerate(zip(ind1_list, ind2_list)):
        #add random factor for exploration
        beta = (1. + 2. * ALPHA) * random.random() - ALPHA
        #add genomes to kids
        ind1_list[position_genome] = (1. - beta) * mixed_tuple_1 + beta * mixed_tuple_2
        ind2_list[position_genome] = beta * mixed_tuple_1 + (1 - beta) * mixed_tuple_2

    ind1.weights = ind1_list[0]
    ind1.stddevs = ind1_list[1]
    ind1.alphas = ind1_list[2]
    
    ind2.weights = ind2_list[0]
    ind2.stddevs = ind2_list[1]
    ind2.alphas = ind2_list[2]

    return ind1, ind2

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

def main():
    global tau, tau_2, beta, ALPHA
    ALPHA = 0.5
    tau = 0.0001
    tau_2 = 0.00001
    beta = 5/ 360 * 2 * np.pi
    #env = environment.Environment(experiment_name = 'Test123', timeexpire = 1000)
    hidden = 10
    population_size = 6
    generations = 5

    env = Environment(experiment_name="test123",
                  playermode="ai",
                  player_controller=player_controller(hidden),
                  speed="fastest",
                  enemymode="static",
                  level=2)

    n_vars = (env.get_num_sensors()+1)*hidden + (hidden + 1)*5
    print(n_vars)           
    max_fitness_per_gen = []
    average = []
    pop = initiate_population(population_size,n_vars, -1, 1)

    for _ in range(generations):

        for individual in pop:
            individual.evaluate(env)

        fitness_list = np.array([individual.fitness for individual in pop])

        new_pop = []
        for _ in range(population_size // 2):
            parent_index_1 = select_tournament(fitness_list, 2)
            parent_index_2 = select_tournament(fitness_list, 2)
            ind1, ind2 = blend_crossover(pop[parent_index_1], pop[parent_index_2])
            ind1.check_and_alter_boundaries()
            ind2.check_and_alter_boundaries()
            
            new_pop.extend([ind1, ind2])

        for individual in new_pop:
            individual.mutate_self()
            individual.check_and_alter_boundaries()
    
        pop = new_pop
        print(len(new_pop))
        print('New generation of degenerates eradicated.')
        print(max(fitness_list))
        average.append(sum(fitness_list))

    print(max_fitness_per_gen)
    plt.plot(average)
    plt.show()

main()
