
# first neural network with keras tutorial
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

# define the keras model
sensors = 10
nodes = 3
model = Sequential()
model.add(Dense(12, input_dim= sensors, activation='relu'))
for _ in range(nodes):
    model.add(Dense(nodes, activation = 'relu'))

model.add(Dense(5, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
x = 1
