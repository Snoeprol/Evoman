# simulate the best player of each generation vs the enemy 5 times

import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np
import os, sys
from os import listdir
from os.path import isfile, join
import statistics
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller
os.putenv('SDL_VIDEODRIVER', 'fbcon')
os.environ["SDL_VIDEODRIVER"] = "dummy"

def takeFirst(elem):
    return elem[0]

def best_ind(runcode):
	# returns the best individual throughout an evolution cycle
	# given a certain runcode
	dir = os.getcwd()

	datapath = dir + "/OutputData/"
	datafiles = [f for f in listdir(datapath) if isfile(join(datapath, f)) if str(runcode) in f]

	stripped_data = []
	max_fitness = 0
	best_individual = 0

	# strip the generation number from the data to enable sorting based on the key
	stripped_data = []
	for f in datafiles:
		stripped_data.append([int(f[19:22].strip(',').strip(' ')), f])

	stripped_data.sort(key=takeFirst)

	sorted_filenames = [i[1] for i in stripped_data]

	# find the best individual
	for f in sorted_filenames:

		df = pd.read_csv(datapath + f)

		individual = df["Fitness"].idxmax()
		if df["Fitness"][individual] > max_fitness:
			max_fitness = df["Fitness"][individual]
			best_individual = individual
			bestdf = df

	weights = bestdf.iloc[best_individual]
	return best_individual, max_fitness, np.array(weights[1:266])

enemy = 1
rank = [0.0659503610347425, 0.6743718281684132, 0.20762736927506664,0.5987394867121654,0.6225654233952975,0.7448307115130878,0.4365745017896734,0.189225964463535,0.5044953136011001,0.5621318338310962]
tour = [0.7119190870137742,0.5876137080725023,0.6710767963116462,0.9285123454226673,0.4172010875390453,0.6268245983870278,0.5086741919295036,0.04757952357627271,0.24641261160854333,0.07921900403119864]

# enemy = 2
# rank = [0.6475525278722439,0.09760541831904068,0.8732192457893112,0.3363246056710042,0.40913528746252537,0.12400987887573223,0.8556354944427789,0.9282807856433142,0.5953200075165063,0.38632605706839074]
# tour = [0.5809718126912192,0.3283451909201691,0.025040675171115745,0.610588012643218,0.6543647096535792,0.9899037515094323,0.7179601520786779,0.10283749095975203,0.030558197699250167,0.3432606756170209]

# enemy = 3
# rank = [0.08411001221521286,0.2451653663046096,0.6221132111928251,0.3566326837129631,0.5569703937982352,0.6105869076143495,0.44134741852091774,0.7738243622876364,0.49517432828864305,0.5834926526529832]
# tour = [0.4631464459873855,0.27132111902229017,0.1082153451542135,0.9536301486536994,0.2763530757899093,0.4783315573211442,0.4688740073077998,0.31016044200712445,0.6346113636328475,0.7936563805049845]

mean_gain = []
energy = []

# run the best individuals  5 times and track the individual gain and player energy
for runcode in rank:
	ind, max_fitness, weights = best_ind(runcode)

	hidden = 10
	gain = []
	for i in range(1):
		env = Environment(experiment_name="test",
		                      playermode="ai",
		                      player_controller=player_controller(hidden),
		                      enemies = [enemy],
		                      speed="fastest",
		                      enemymode="static",
		                      level=2)

		f, p, e, t = env.play(weights)
		gain.append(p - e)
	energy.append(p)
	mean_gain.append(np.mean(gain))

# save the data
energydf = pd.DataFrame(energy)
energydf.to_csv("energy" + str(enemy) + "rank.csv")

df = pd.DataFrame(mean_gain)
df.to_csv("sim" + str(enemy) + "tour.csv")

plt.boxplot(mean_gain)
plt.show()
