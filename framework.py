import sys
sys.path.insert(0, 'evoman')
from environment import Environment
import numpy as np
import scipy.stats as sp
import random
from demo_controller import player_controller
import matplotlib.pyplot as plt

min_weight = -1
max_weight = 1

def initiate_population(size, variables, min_weight, max_weight):
    ''' Initiate a population of individuals with variables amount of parameters unfiformly 
    chosen between min_weight and max_weight'''
    population = []
    alphas = []

    # random values between the standard deviation of a uniform distribution between [-1, 1]
    stddevs = [1/np.sqrt(6)] * size

    for _ in range(int(size * (size - 1) / 2)):
        alphas.append(np.random.uniform(-np.pi, np.pi))
    for _ in range(size):
        population.append(np.random.rand(variables) * (max_weight - min_weight) +  min_weight)

    return np.array([population, stddevs, alphas])

def generate_individual(variables, min_weight, max_weight):
    '''Returns individual of the population with variables amount of parameters uniformly chosen
    between min_weight and max_weight'''
    individual = np.random.rand(variables) * (max_weight - min_weight) +  min_weight
    return individual

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
    hidden = 30
    population_size = 10
    generations = 50
    env = Environment(experiment_name="test123",
				  playermode="ai",
				  player_controller=player_controller(hidden),
			  	  speed="fastest",
				  enemymode="static",
				  level=1)

    n_vars = (env.get_num_sensors()+1)*hidden + (hidden + 1)*5           
    max_fitness_per_gen = []
    average = []
    population = initiate_population(population_size,n_vars, -1, 1)
    

    for _ in range(generations):
        fitness_list = evaluate_population(env, population[0])
        generate_next_generation(population, fitness_list)
        max_fitness_per_gen.append(max(fitness_list))
        print('New generation of degenerates eradicated.')
        print(max(fitness_list))
        average.append(sum(fitness_list))

    print(max_fitness_per_gen)
    plt.plot(average)
    plt.show()

main()
