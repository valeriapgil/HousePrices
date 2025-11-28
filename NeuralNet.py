import numpy as np

class NeuralNet:

  #Constructor
  def __init__(self, layers, epochs=1000, learning_rate=0.001, momentum=0.9, function='relu', validation_split=0.2):
    self.L = len(layers) #Number of layers
    self.n = layers.copy() #Array with number of units(neurons) in each layer including the input and the output layers
    self.epochs = epochs #Number of epochs
    self.learning_rate = learning_rate #Learning rate
    self.momentum = momentum #Momentum
    self.function = function #Selected activation function
    
    #Definition of the possible activation functions
    self.activation_functions = {
      "sigmoid": self.sigmoid,
      "relu": lambda x: np.maximum(0, x),
      "linear": lambda x: x,
      "tanh": lambda x: np.tanh(x)
    }
    self.fact = self.activation_functions.get(self.function, self.sigmoid)

    #Definition of the derivatives of the possible activation functions
    self.derivative_functions = {
      "sigmoid": lambda x: self.sigmoid(x) * (1 - self.sigmoid(x)),
      "relu": lambda x: np.where(x > 0, 1, 0),
      "linear": lambda x: np.ones_like(x),
      "tanh": lambda x: 1 - np.tanh(x) ** 2
    }
    self.fact_der = self.derivative_functions.get(self.function, lambda x: self.sigmoid(x) * (1 - self.sigmoid(x)))

    
    self.validation_split = validation_split #Percentage of data for validation
  
    self.h = [] #Array of arrays for the fields
    for lay in range(self.L):
      self.h.append(np.zeros(layers[lay]))
      
    self.xi = [] #Array of arrays for the activations
    for lay in range(self.L):
      self.xi.append(np.zeros(layers[lay])) 

    self.w = [] #Array of matrices for the weights
    self.w.append(np.zeros((1, 1)))
    for lay in range(1, self.L):
        self.w.append(np.random.randn(layers[lay], layers[lay - 1]) * np.sqrt(2. / layers[lay - 1]))

    self.theta = [] #Array of arrays for the thresholds
    for lay in range(self.L):
      self.theta.append(np.random.randn(layers[lay]) * np.sqrt(2. / layers[lay]))
  
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

    self.training_error = [] #Array for saving training errors.
    self.validation_error = [] #Array for saving validation errors.


  def fit(self, X: np.ndarray, y: np.ndarray):
    """
      fit

      This function splits the dataset with the given X and y into training and validation sets using the 
      validation_split parameter. The training set is used for learning, while the validation set is used 
      to monitor performance on unseen data. 
      
      It performs the training process over a specified number of epochs. In each epoch, the neural network
      is trained on the entire training set. For every sample: it is passed through the network to obtain a 
      prediction, the error is calculated, the error gradients are propagated backward, and finally the 
      model’s internal weights and thresholds are adjusted to reduce the error. After each epoch, the 
      function calculates and stores the Mean Squared Error for both the training set and the validation set.

   """
    n_samples = X.shape[0]
    n_val = int(n_samples * self.validation_split)
  
    indices = np.random.permutation(n_samples)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
            
    for epoch in range(self.epochs):
        # loop through all patterns in random order
      for i in np.random.permutation(X_train.shape[0]):
        x_patron = X_train[i].reshape(-1)
        y_patron = y_train[i].reshape(-1)
        self.feed_foward(x_patron)
        self.back_propagation_errors(y_patron)
        self.update_weights_and_thresholds()
      
      train_error = self.mean_squared_error(X_train, y_train)
      val_error = self.mean_squared_error(X_val, y_val)
      self.training_error.append(train_error)
      self.validation_error.append(val_error)

      if epoch % 100 == 0:
        print(f"Epoch {epoch}/{self.epochs} - Training Error: {train_error:.6f} - Validation Error: {val_error:.6f}")

  def predict(self, x: np.ndarray):
    """
      Predict 

      After the model is trained, this function performs feed-forward propagation for every sample in the 
      dataset X to obtain a prediction.
    
    """
    predictions = []
    for i in range(x.shape[0]):
      self.feed_foward(x[i]) #Feed forward for each input sample
      predictions.append(self.xi[-1]) #Output layer activations
    return np.array(predictions)
  
  def loss_epochs(self):
    """
      loss_epochs

      This function returns two arrays: one containing the training error values and the other containing
      the validation error values computed during the fit function.

    """
    return np.array(self.training_error), np.array(self.validation_error)

  def feed_foward(self, x: np.ndarray):
    """
      feed_forward

      This function first computes the fields and then the activations (using the chosen activation
      function) for each layer of the neural network, using the thresholds and the weights.

    """
    self.xi[0] = x
    for lay in range(1, self.L):
      self.h[lay] = np.dot(self.w[lay], self.xi[lay - 1]) - self.theta[lay]
      self.xi[lay] = self.fact(self.h[lay])

  def back_propagation_errors(self, y: np.ndarray):
    """
      back_propagation_errors

      Once the feed-forward propagation is completed and the predictions are obtained, this function 
      calculates the error of the output layer and then propagates it backward. The errors are saved
      and later used to update the weights and thresholds.

    """
    self.delta[-1] = self.fact_der(self.h[-1]) * (self.xi[-1] - y)
    for lay in range(self.L - 2, 0, -1):
      self.delta[lay] = self.fact_der(self.h[lay]) * np.dot(self.w[lay + 1].T, self.delta[lay + 1])

  def update_weights_and_thresholds(self):
    """
      update_weights_and_thresholds

      this function updates the network’s weights and thresholds using the gradient descent algorithm 
      with momentum. It uses the errors computed during backpropagation, while storing prior 
      updates to incorporate the momentum effect.

    """
    for lay in range(1, self.L):
      self.d_w[lay] = -self.learning_rate * np.outer(self.delta[lay], self.xi[lay - 1]) + self.momentum * self.d_w_prev[lay]
      self.d_w_prev[lay] = self.d_w[lay]
      self.w[lay] += self.d_w[lay]

      self.d_theta[lay] = self.learning_rate * self.delta[lay] + self.momentum * self.d_theta_prev[lay]
      self.d_theta_prev[lay] = self.d_theta[lay]
      self.theta[lay] += self.d_theta[lay]

  def mean_squared_error(self, X: np.ndarray, y: np.ndarray):
    """
    mean_squared_error

      This function computes the Mean Squared Error between the predicted and actual values of each sample,
      and returns the average.
      
    """
    error = 0.0
    for i in range(X.shape[0]):
      self.feed_foward(X[i]) 
      y_pred = self.xi[-1] 
      error += np.sum((y_pred - y[i]) ** 2)
    total_error = error / X.shape[0]
    return total_error

  def sigmoid(self, x):
    """
    Sigmoid

    Definition of sigmoid function.
    """
    return 1 / (1 + np.exp(-x))


# layers = [4, 9, 5, 1]
# nn = NeuralNet(layers)

# print("L = ", nn.L, end="\n")
# print("n = ", nn.n, end="\n")

# print("xi = ", nn.xi, end="\n")
# print("xi[0] = ", nn.xi[0], end="\n")
# print("xi[1] = ", nn.xi[0], end="\n")

# print("wh = ", nn.w, end="\n")
# print("wh[1] = ", nn.w[1], end="\n")
