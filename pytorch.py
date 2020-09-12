import torch
import numpy as np
from torch import nn

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(784, 256)
        self.hidden_2 = nn.Linear(256, 256)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(256, 10)
        
        # Define sigmoid activation and softmax output 
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.hidden_2(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)
        
        return x


network = Network()
x = torch.tensor(1)
print(network.state_dict())
#snetwork.state_dict()['features.0.weight'].data.copy_(x)
weights = 256 * 784
list_weights = []
list_bias = []
for i in range(weights):
    list_weights.append(i)
for i in range(256):
    list_bias.append()
z = np.array(list_weights)
z = np.reshape(z, (256,784))
z = torch.from_numpy(z)
params_total = 0
for item in network.state_dict():
    print(network.state_dict()[item].size())

    network.state_dict()[item].data.copy_(z)
weights = np.array(256, 784)
y = network.state_dict()['hidden.weight'].size()
print(y)
for tensor in y:
    x = 1
parent_model = Network()
model_to_update = Network()

update_dict = {k: v for k, v in parent_model.state_dict().items() if k in model_to_update.state_dict()}

print(len(model_to_update.state_dict()))
print(len(update_dict))

#for param in network.parameters():
#    for number in param:
#        for low in number:
#                print(low)
print(len(network.hidden.bias))
x = list(network.hidden.parameters())
#print(network.hidden[0].weight)
for i in range(10):
    
    network.hidden.bias[i] = i
#for i in range()
y = torch.cat([w.view(-1) for w in network.parameters()])
print(y.size())
for param in y:
    param = torch.tensor(1.1)

z = network.parameters()
p = torch.cat([w.view(-1) for w in network.parameters()])
a = torch.randn(256, dtype=torch.double)
print(a)
print(a.size())
#replace = FloatTensor.
#print(network.hidden.bias.size())

#network.hidden.bias = torch.FloatTensor([[1], [1]])
#print(network.hidden.bias.indices())
x = len(network.hidden.bias)
for i in range(x):
    network.hidden.bias[int(i)] = np.random.randn(1)
print(len(network.hidden.bias[0]))
#network.hidden.bias = np.array(np.random.randn(256))
print(len(network.hidden.bias))
print(len(network.hidden.weight[0]))
network.parameters