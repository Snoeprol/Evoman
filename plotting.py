# create the data used to create the graphs of the fitness values

import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import statistics
import csv

def takeFirst(elem):
    return elem[0]

def mean_max(runcode):
	# returns the 
	dir = os.getcwd()

	datapath = dir + "/OutputData/"
	datafiles = [f for f in listdir(datapath) if isfile(join(datapath, f)) if str(runcode) in f]

	stripped_data = []
	for f in datafiles:
		stripped_data.append([int(f[19:22].strip(',').strip(' ')), f])

	stripped_data.sort(key=takeFirst)

	sorted_filenames = [i[1] for i in stripped_data]

	max_fitness = []
	mean_fitness = []
	for f in sorted_filenames:
		df = pd.read_csv(datapath + f)
		max_fitness.append(max(df["Fitness"]))
		mean_fitness.append(np.mean(df["Fitness"]))

	generation = stripped_data[-1][0]
	
	return mean_fitness, max_fitness, generation

enemy = 1
rank = [0.0659503610347425, 0.6743718281684132, 0.20762736927506664,0.5987394867121654,0.6225654233952975,0.7448307115130878,0.4365745017896734,0.189225964463535,0.5044953136011001,0.5621318338310962]
tour = [0.7119190870137742,0.5876137080725023,0.6710767963116462,0.9285123454226673,0.4172010875390453,0.6268245983870278,0.5086741919295036,0.04757952357627271,0.24641261160854333,0.07921900403119864]

# enemy = 2
# rank = [0.6475525278722439,0.09760541831904068,0.8732192457893112,0.3363246056710042,0.40913528746252537,0.12400987887573223,0.8556354944427789,0.9282807856433142,0.5953200075165063,0.38632605706839074]
# tour = [0.5809718126912192,0.3283451909201691,0.025040675171115745,0.610588012643218,0.6543647096535792,0.9899037515094323,0.7179601520786779,0.10283749095975203,0.030558197699250167,0.3432606756170209]

# enemy = 3
# rank = [0.08411001221521286,0.2451653663046096,0.6221132111928251,0.3566326837129631,0.5569703937982352,0.6105869076143495,0.44134741852091774,0.7738243622876364,0.49517432828864305,0.5834926526529832]
# tour = [0.4631464459873855,0.27132111902229017,0.1082153451542135,0.9536301486536994,0.2763530757899093,0.4783315573211442,0.4688740073077998,0.31016044200712445,0.6346113636328475,0.7936563805049845]

runcodescombined = [rank, tour]
run = 0
for runcodes in runcodescombined:
	meanfs = []
	maxfs = []
	for runcode in runcodes:
		meanf, maxf, g = mean_max(runcode) 
		meanfs.append(meanf)
		maxfs.append(maxf)



	avg_max = []
	std_max = []

	avg_mean = []
	std_mean = []
	for i in range(g + 1):

		max_values = []

		mean_values = []
		for j in range(len(maxfs)):

			max_values.append(maxfs[j][i])

			mean_values.append(meanfs[j][i])

		std_max.append(statistics.stdev(max_values))
		avg_max.append(np.mean(max_values))
		std_mean.append(statistics.stdev(mean_values))
		avg_mean.append(np.mean(mean_values))


	generations = np.linspace(0,g, g + 1)

	plt.title("Results enemy {}".format(enemy))

	if run == 0:
		plt.errorbar(generations, avg_max, yerr = std_max, label='rank max')
		plt.errorbar(generations, avg_mean, yerr = std_mean, label='rank mean')

		d = {"max":avg_max, "maxstd":std_max, "mean":avg_mean, "meanstd":std_mean}
		df = pd.DataFrame(d)
		df.to_csv("rank{}.csv".format(enemy))
		d = {"max":maxfs, "mean":meanfs}
		df = pd.DataFrame(d)
		df.to_csv("max_mean_rank{}.csv".format(enemy))
	else:
		plt.errorbar(generations, avg_max, yerr = std_max, label='tournament max')
		plt.errorbar(generations, avg_mean, yerr = std_mean, label='tournament mean')

		d = {"max":avg_max, "maxstd":std_max, "mean":avg_mean, "meanstd":std_mean}
		df = pd.DataFrame(d)
		df.to_csv("tour{}.csv".format(enemy))
		d = {"max":maxfs, "mean":meanfs}
		df = pd.DataFrame(d)
		df.to_csv("max_mean_tour{}.csv".format(enemy                 ))
	run += 1
plt.legend()
# plt.show()