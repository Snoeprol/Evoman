from tensorflow import keras
from keras import models

class Controller():

    def __init__(self, nodes = 10,layers = 2):
        model = keras.models.Sequential()
        model.add(keras.Input(shape=(20,)))
        for i in range(layers):
            model.add(keras.layers.Dense(nodes, activation='relu'))
        # Now the model will take as input arr
        # and output arrays of shape (None, 3ays of shape (None, 16)2).
        # Note that after the first layer, you don't need to specify
        # the size of the input anymore:
        model.add(keras.layers.Dense(5))
        print(model.output_shape)
        def control(self, inputs):

            inputs = (inputs-min(inputs))/float((max(inputs)-min(inputs)))


            return []

controller = Controller()
