# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 10:08:57 2020

@author: igas
"""
import os
import pandas as pd
import scipy.stats as sc

os.chdir("boxplotdata")

sim1rank = pd.read_csv("sim1rank.csv").iloc[:,1]
sim1tour = pd.read_csv("sim1tour.csv").iloc[:,1]
sim2rank = pd.read_csv("sim2rank.csv").iloc[:,1]
sim2tour = pd.read_csv("sim2tour.csv").iloc[:,1]
sim3rank = pd.read_csv("sim3rank.csv").iloc[:,1]
sim3tour = pd.read_csv("sim3tour.csv").iloc[:,1]

print(sim1rank, sim1tour)

sim1test_t, sim1test_p = sc.ttest_ind(sim1rank, sim1tour, equal_var = False)
sim2test_t, sim2test_p = sc.ttest_ind(list(sim2rank), list(sim2tour), equal_var = False)
sim3test_t, sim3test_p = sc.ttest_ind(list(sim3rank), list(sim3tour), equal_var = False)

print(sim1test_p)
print(sim2test_p)
print(sim3test_p)