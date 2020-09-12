import numpy as np
import time
#controller = Controller()

hidden = 10
layers = 5
output = 5
input = 20
params = np.random.randn((layers - 1) * hidden + (layers -1 ) * (hidden **2) + (hidden + 1) * input + (hidden + 1) * output)
input = np.random.randn(20)

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('Function time ' + method.__name__ + ': ' + str(round((te - ts) * 1000,4)) + 'ms')
        return result
    return timed

def relu(x):
   return np.maximum(0,x)

def softmax(X):
    expo = np.exp(X)
    expo_sum = np.sum(np.exp(X))
    return expo/expo_sum

def forward(input,params, neurons_per_layer, layers):
    ''' Neural network: input types: list, np.array, int, int
    returns the output given the weight size params: layers ( hidden * ( 1 + input)) + hidden * (output + 1)'''


    signal_length = len(input)
    start = 0
    finish = signal_length * neurons_per_layer
    
    # First layer
    A =  np.reshape(params[start:finish], (neurons_per_layer, signal_length))
    input = np.dot(A, input)
    start = finish
    finish += hidden
    input += params[start:finish]
    input = relu(input)

    # Middle layers
    for layer in range(layers - 1):
        finish = start + neurons_per_layer * neurons_per_layer

        # Define matrix
        A =  np.reshape(params[start:finish], (neurons_per_layer, neurons_per_layer))
        input = np.dot(A, input)
        input += np.array(params[finish:finish + neurons_per_layer])
        input = relu(input)
        start = finish + neurons_per_layer
        finish = neurons_per_layer * neurons_per_layer

    # Last layer
    output_size = 5
    size = neurons_per_layer * output_size
    y = params[start:(start + size)]
    A = np.reshape(params[start:start + size], (neurons_per_layer, output_size))
    input = np.dot(input, A) 
    input += np.array(params[start + size:start + size + output_size])
    input = softmax(input)

    return input

forward(input, params, hidden, layers)

