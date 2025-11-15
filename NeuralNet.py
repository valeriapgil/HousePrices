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
      "relu": lambda x: np.maximum(0, x),
      "linear": lambda x: x,
      "tanh": lambda x: np.tanh(x)
    }
    self.fact = self.activation_functions.get(self.function, self.sigmoid)
    self.derivative_functions = {
      "sigmoid": lambda x: self.sigmoid(x) * (1 - self.sigmoid(x)),
      "relu": lambda x: np.where(x > 0, 1, 0),
      "linear": lambda x: np.ones_like(x),
      "tanh": lambda x: 1 - np.tanh(x) ** 2
    }

    self.fact_der = self.derivative_functions.get(self.function, lambda x: self.sigmoid(x) * (1 - self.sigmoid(x)))
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
    
    self.delta = [] #Array of arrays for the propagation errors
    for lay in range(self.L):
      self.delta.append(np.zeros(layers[lay]))

    self.d_w = [] #Array of matrices for the changes of weights
    self.d_w.append(np.zeros((1, 1)))
    for lay in range(1, self.L):
      self.d_w.append(np.zeros((layers[lay], layers[lay - 1])))
    
    self.d_theta = [] #Array of arrays for the changes of thresholds
    for lay in range(self.L):
      self.d_theta.append(np.zeros(layers[lay])) 
    
    self.d_w_prev = [] #Array of matrices for the previous changes of weights
    self.d_w_prev.append(np.zeros((1, 1)))
    for lay in range(1, self.L):
      self.d_w_prev.append(np.zeros((layers[lay], layers[lay - 1])))

    self.d_theta_prev = [] #Array of arrays for the previous changes of thresholds
    for lay in range(self.L):
      self.d_theta_prev.append(np.zeros(layers[lay]))

  def feed_foward(self, x: np.ndarray):
    self.xi[0] = x
    for lay in range(1, self.L):
      self.h[lay] = np.dot(self.w[lay], self.xi[lay - 1]) - self.theta[lay]
      self.xi[lay] = self.fact(self.h[lay])

  def back_propagation_errors(self, y: np.ndarray):
    self.delta[-1] = self.fact_der(self.h[-1]) * (self.xi[-1] - y) #L-1?
    for lay in range(self.L - 2, 0, -1): #L-2?
      self.delta[lay] = self.fact_der(self.h[lay]) * np.dot(self.w[lay + 1].T, self.delta[lay + 1])

  def update_weights_and_thresholds(self):
    for lay in range(1, self.L):
      self.d_w[lay] = -self.learning_rate * np.outer(self.delta[lay], self.xi[lay - 1]) + self.momentum * self.d_w_prev[lay]
      self.d_w_prev[lay] = self.d_w[lay]
      self.w[lay] += self.d_w[lay]

      self.d_theta[lay] = self.learning_rate * self.delta[lay] + self.momentum * self.d_theta_prev[lay]
      self.d_theta_prev[lay] = self.d_theta[lay]
      self.theta[lay] += self.d_theta[lay]

  def sigmoid(self, x):
    return 1 / (1 + np.exp(-x))


layers = [4, 9, 5, 1]
nn = NeuralNet(layers)

print("L = ", nn.L, end="\n")
print("n = ", nn.n, end="\n")

print("xi = ", nn.xi, end="\n")
print("xi[0] = ", nn.xi[0], end="\n")
print("xi[1] = ", nn.xi[0], end="\n")

print("wh = ", nn.w, end="\n")
print("wh[1] = ", nn.w[1], end="\n")
