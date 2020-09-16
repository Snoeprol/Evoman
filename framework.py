import sys
sys.path.insert(0, 'evoman')
from environment import Environment
import numpy as np
import scipy.stats as sp
import random
from demo_controller import player_controller
import matplotlib.pyplot as plt
from covariance import mutate 

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


def generate_individual(variables, min_weight, max_weight):
    '''Returns individual of the population with variables amount of parameters uniformly chosen
    between min_weight and max_weight'''
    individual = np.random.rand(variables) * (max_weight - min_weight) +  min_weight
    return individual

def check_and_alter_boundaries(ind1):
    #check if all the weight values are between -1, 1
    if (min(ind1[0]) < -1) or (max(ind1[0]) > 1):
        #not the case, change the values to max allowed value
        for i in range(ind1[0]):
            if ind1[0][i] < -1:
                ind1[0][i] = -1
            if ind1[0][i] > 1:
                ind1[0][i] = 1
                
     if (min(ind1[2]) < - math.pi) or (max(ind1[2]) > math.pi):
        for i in range(ind1[2]):
            if ind1[2][i] < -math.pi:
                ind1[2][i] = -math.pi
            if ind1[2][i] > math.pi:
                ind1[2][i] = math.pi
                
    if (min(ind1[3]) < - math.pi) or (max(ind1[3]) > math.pi):
        for i in range(ind1[3]):
            if ind1[3][i] < -math.pi:
                ind1[3][i] = -math.pi
            if ind1[3][i] > math.pi:
                ind1[3][i] = math.pi
                
    return ind1

        
def Blend_Crossover(ind1, ind2):
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
    
    

def replace_portion_random(percentage, fitness_list, population, min_weight, max_weight):
    '''Replaces the worst portion of the population with randomly initialized individuals'''
    replaced = int(percentage * len(fitness_list) / 100)
    worst_indices = select_worst(fitness_list, replaced)
    for i in worst_indices:
        population[i] = generate_individual(len(population[i]), min_weight, max_weight)

def generate_test_fitness(individuals):
    '''Generate a list of random fitnesses to test'''
    return np.random.rand(individuals)

    
def select_worst(fitness_list, n):
    '''Return indices of n worst individuals'''
    return fitness_list.argsort()[:n]

def select_best(fitness_list, n):
    '''Returns indices of n best individuals'''
    return (-fitness_list).argsort()[:n]

def select_individuals_fitness(fitness_list, portion):
    '''Selects individuals, giving individuals with a lower fitness
    a higher chance to die.'''

    total = calculate_fitness(fitness_list)
    
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

def crossover_population(population, fitness_list, percentage):
    '''Generates k-point crossover in population with certain pentage of the population
    letting the fittest pairs create children''' 
    individuals = percentage * len(population) / 100
    individuals = int(individuals) if ((int(individuals) % 2) == 0) else int(individuals) + 1
    best_indices = select_best(fitness_list, individuals)

    # Let the best pair create new individuals
    for i, value in enumerate(best_indices):
        if i % 2 == 0:
            k_point_crossover(population[i], population[i + 1])

def k_point_crossover(parent_1, parent_2):
    '''Given 2 individuals, creates 2 children with k_point crossover,
    where k is picked from an absolute normal distribution with average of 0 
    (set to 1 if actually 0) and st. dev. amount of traits / 5'''
    traits = len(parent_1)
    k = abs(np.random.normal(0, traits/ float(5)))
    if k > traits:
        k = traits
    if k < 1:
        k = 1
    k = int(k)
    
    crossover_points = random.sample(range(1, traits), k)
    crossover_points.append(traits) 
    crossover_points.sort()

    child_1 = np.array([])
    child_2 = np.array([])
    i = 0
    for index, j in enumerate(crossover_points):
        if index % 2 == 0:
            child_1 = np.concatenate((child_1, parent_1[i:j]))
            child_2 = np.concatenate((child_2, parent_2[i:j]))
        else:
            child_1 = np.concatenate((child_1, parent_2[i:j]))
            child_2 = np.concatenate((child_2, parent_1[i:j]))
        i = j
    
    parent_1 = child_1
    parent_2 = child_2

    return child_1, child_2


def mutate_population(population, fitness_list):
    for individual in population:
        mutate_individual(individual)

def mutate_individual(individual):
    '''Given an individual, generates on average 1 mutation in a trait
    with a st. dev. of 1 from a normal distribution'''
    matuation_chance = 1 - 1/len(individual)
    for trait in individual:
        if np.random.rand() > matuation_chance:
            mutation_size = np.random.normal(0, 1)
            # Boundaries
            if min_weight < trait + mutation_size < max_weight:
                trait = trait + mutation_size
            
            elif trait + mutation_size < min_weight:
                trait = min_weight
            else:
                trait = max_weight 

def generate_next_generation(population, fitness_list):
    replacement_percentage = 20
    replace_portion_random(20, fitness_list, population, min_weight, max_weight)
    crossover_population(population, fitness_list, 100 - replacement_percentage)
    mutate_population(population, fitness_list)

def evaluate_population(env, population):
    return np.array(list(map(lambda individual: simulation(env,individual), population)))

def simulation(env,x):
    f,p,e,t = env.play(x)
    return f

def main():
    #env = environment.Environment(experiment_name = 'Test123', timeexpire = 1000)
    hidden = 10
    population_size = 10
    generations = 2

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
    population = initiate_population(population_size,n_vars, -1, 1)


    print(population[0].weights[0])
    population[0].mutate_self()
    print(population[0].weights[0])
    for _ in range(generations):
        for individual in population:
            individual.evaluate(env)

        fitness_list = np.array([individual.fitness for individual in population])

        for individual in population:
            individual.mutate_self()


        generate_next_generation(population, fitness_list)
        max_fitness_per_gen.append(max(fitness_list))
        print('New generation of degenerates eradicated.')
        print(max(fitness_list))
        average.append(sum(fitness_list))

    print(max_fitness_per_gen)
    plt.plot(average)
    plt.show()

main()
