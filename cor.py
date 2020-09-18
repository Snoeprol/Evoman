import matplotlib.pyplot as plt
import numpy

for i in range(100):
    x = numpy.random.multivariate_normal([0,0], [[4,150],[150,1]])
    plt.plot(x[0], x[1], '.')


plt.show()