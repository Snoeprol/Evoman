import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np 

enemy = 1
sns.set(font_scale=1.65)
plt.figure(figsize=(8, 6))
filenames = ["sim" + str(enemy) + "rank.csv", "sim" + str(enemy) + "tour.csv"]

d = pd.read_csv(filenames[0], names = ["EA2"], skiprows=1)
df = pd.read_csv(filenames[1], names = ["EA1"], skiprows = 1)
d["EA1"] = df["EA1"]

sns.boxplot(data = d, orient="v")
plt.title("Individual gain versus enemy {}".format(enemy))
plt.ylabel("Individual Gain [-]")
plt.xlabel("Type of selection")
plt.show()


