import matplotlib.pyplot as plt
import matplotlib as mpl
year = []
name_2 = []
name_3 = []
mpl.rcParams['figure.dpi'] = 300
with open(r'C:\Users\mario\OneDrive\Documenten\CompSci\Evolutionary_computing\EvoMan\evoman_framework\istherecorrelation.csv') as f:
    data = f.readlines()
    for line in data[1:]:
        y = line.replace(';', ',')
        y = y.split(',')
        year.append(int(y[0]))
        name_2.append(int(y[1]))
        name_3.append(int(y[2]))
#plt.plot(year, name_2, label = 'Unknown')
plt.plot(year, name_3, label = 'Beer consumption')
plt.ylabel('Beer consumption in the Netherlands [hectoliters]')
plt.xlabel('Year')
plt.xlim([2006,2018])
plt.legend()
plt.show()
    