# create the plots using the data created in plotting.py

import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np 
sns.set(font_scale=1.65)
plt.figure(figsize=(8, 6))
enemy = 3

df = pd.read_csv("rank" + str(enemy) + ".csv")
plt.title("Fitness progression versus enemy {}".format(enemy))
alpha = 0.2
g = np.linspace(0, 49, 50)
sns.lineplot(x = g, y = "max", data = df, color = "blue", label="EA2 max")
yp = df["max"] - df["maxstd"]
sns.lineplot(x = g, y = yp, data = df, color='blue', alpha = alpha)
ym = df["max"] + df["maxstd"]
sns.lineplot(x = g, y = ym, data = df, color='blue', alpha = alpha)

plt.fill_between(g, yp, ym, color = 'blue', alpha = alpha)


sns.lineplot(x = g, y = "mean", data = df, color = "green", label="EA2 mean")
yp = df["mean"] - df["meanstd"]
sns.lineplot(x = g, y = yp, data = df, color='green', alpha = alpha)
ym = df["mean"] + df["meanstd"]
sns.lineplot(x = g, y = ym, data = df, color='green', alpha = alpha)

plt.fill_between(g, yp, ym, color = 'green', alpha = alpha)

df = pd.read_csv("tour" + str(enemy) + ".csv")


sns.lineplot(x = g, y = "max", data = df, color = "red", label="EA1 max")
yp = df["max"] - df["maxstd"]
sns.lineplot(x = g, y = yp, data = df, color='red', alpha = alpha)
ym = df["max"] + df["maxstd"]
sns.lineplot(x = g, y = ym, data = df, color='red', alpha = alpha)

plt.fill_between(g, yp, ym, color = 'red', alpha = alpha)


sns.lineplot(x = g, y = "mean", data = df, color = "orange", label="EA1 mean")
yp = df["mean"] - df["meanstd"]
sns.lineplot(x = g, y = yp, data = df, color='orange', alpha = alpha)
ym = df["mean"] + df["meanstd"]
sns.lineplot(x = g, y = ym, data = df, color='orange', alpha = alpha)

plt.fill_between(g, yp, ym, color = 'orange', alpha = alpha)

plt.ylabel("Fitness [-]")
plt.xlabel("Generation [-]")
plt.legend()
plt.show()