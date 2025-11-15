import numpy as np

class NeuralNet:
  def __init__(self, layers, epochs=1000, learning_rate=0.01, momentum=0.9, function='sigmoid', validation_split=0.2):
    self.L = len(layers) #Number of layers
    self.n = layers.copy() #Array with number of units(neurons) in each layer including the input and the output layers
    self.epochs = epochs
    self.learning_rate = learning_rate
    self.momentum = momentum
    self.function = function
    self.activation_functions = {
      "sigmoid": self.sigmoid,
      "relu": self.relu,
      "linear": self.linear,
      "tanh": self.tanh
    }
    self.act_fun = self.activation_functions.get(self.function, self.sigmoid) 
    self.validation_split = validation_split
  
    self.h = [] #Array of arrays for the fields
    for lay in range(self.L):
      self.h.append(np.zeros(layers[lay])) #*******
      
    self.xi = [] #Array of arrays for the activations
    for lay in range(self.L):
      self.xi.append(np.zeros(layers[lay])) 

    self.w = [] #Array of matrices for the weights
    self.w.append(np.zeros((1, 1)))
    for lay in range(1, self.L):
      self.w.append(np.zeros((layers[lay], layers[lay - 1])))

    self.theta = [] #Array of arrays for the thresholds
    for lay in range(self.L):
      self.theta.append(np.zeros(layers[lay]))

  def feed_foward(self, x: np.ndarray):
    self.xi[0] = x
    for lay in range(1, self.L):
      self.h[lay] = np.dot(self.w[lay], self.xi[lay - 1]) - self.theta[lay]
      self.xi[lay] = self.act_fun(self.h[lay])


  def sigmoid(self, x):
    return 1 / (1 + np.exp(-x))
  
  def relu(self, x):
    return np.maximum(0, x)
  
  def linear(self, x):
    return x
  
  def tanh(self, x):
    return np.tanh(x)
     


layers = [4, 9, 5, 1]
nn = NeuralNet(layers)

print("L = ", nn.L, end="\n")
print("n = ", nn.n, end="\n")

print("xi = ", nn.xi, end="\n")
print("xi[0] = ", nn.xi[0], end="\n")
print("xi[1] = ", nn.xi[0], end="\n")

print("wh = ", nn.w, end="\n")
print("wh[1] = ", nn.w[1], end="\n")
