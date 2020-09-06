from tensorflow import keras
from tensorflow.keras import layers
import sys, os
sys.path.insert(0, 'evoman')
from environment import Environment

env = Environment(experiment_name=experiment_name,
                  enemymode='static',
                  speed="normal",
                  sound="on",
                  fullscreen=True,
                  playermode='human')
env.update_parameter('enemies', [en])
env.play()

class MyModel(tf.keras.Model):

    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)
        self.dropout = tf.keras.layers.Dropout(0.5)

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        if training:
            x = self.dropout(x, training=training)
        return self.dense2(x)

model = MyModel()

def get_action_from_network(state, network):

