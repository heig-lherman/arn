import numpy as np

class MLP:
    '''
    This code was adapted from:
    https://rolisz.ro/2013/04/18/neural-networks-in-python/
    '''
    def __tanh(self, x):
        '''Hyperbolic tangent function'''
        return np.tanh(x)

    def __tanh_deriv(self, a):
        '''Hyperbolic tangent derivative'''
        return 1.0 - a**2

    def __logistic(self, x):
        '''Sigmoidal function'''
        return 1.0 / (1.0 + np.exp(-x))

    def __logistic_derivative(self, a):
        '''sigmoidal derivative'''
        return a * ( 1 - a )
    
    def __init__(self, layers, activation='tanh'):
        '''
        :param layers: A list containing the number of units in each layer.
        Should be at least two values
        :param activation: The activation function to be used. Can be
        "logistic" or "tanh"
        '''
        self.n_inputs = layers[0]                               # Number of inputs (first layer)
        self.n_outputs = layers[-1]                             # Number of ouputs (last layer)
        self.layers = layers
                                                                # Activation function
        if activation == 'logistic':
            self.activation = self.__logistic
            self.activation_deriv = self.__logistic_derivative
        elif activation == 'tanh':
            self.activation = self.__tanh
            self.activation_deriv = self.__tanh_deriv

        self.init_weights()                                     # Initialize the weights of the MLP
        
    def init_weights(self):
        '''
        This function creates the matrix of weights and initialiazes their values to small values
        '''
        self.weights = []                                       # Start with an empty list
        self.delta_weights = []
        for i in range(1, len(self.layers) - 1):                # Iterates through the layers
                                                                # np.random.random((M, N)) returns a MxN matrix
                                                                # of random floats in [0.0, 1.0).
                                                                # (self.layers[i] + 1) is number of neurons in layer i plus the bias unit
            self.weights.append((2 * np.random.random((self.layers[i - 1] + 1, self.layers[i] + 1)) - 1) * 0.25)
                                                                # delta_weights are initialized to zero
            self.delta_weights.append(np.zeros((self.layers[i - 1] + 1, self.layers[i] + 1)))
                                                                # Append a last set of weigths connecting the output of the network
        self.weights.append((2 * np.random.random((self.layers[i] + 1, self.layers[i + 1])) - 1) * 0.25)
        self.delta_weights.append(np.zeros((self.layers[i] + 1, self.layers[i + 1])))
   
    def fit(self, data_train, data_test=None, learning_rate=0.1, momentum=0.7, epochs=100):
        '''
        Online learning.
        :param data_train: A tuple (X, y) with input data and targets for training
        :param data_test: A tuple (X, y) with input data and targets for testing
        :param learning_rate: parameters defining the speed of learning
        :param epochs: number of times the dataset is presented to the network for learning
        '''
        X = np.atleast_2d(data_train[0])                        # Inputs for training
        temp = np.ones([X.shape[0], X.shape[1]+1])              # Append the bias unit to the input layer
        temp[:, 0:-1] = X                                       
        X = temp                                                # X contains now the inputs plus a last column of ones (bias unit)
        y = np.array(data_train[1])                             # Targets for training
        error_train = np.zeros(epochs)                          # Initialize the array to store the error during training (epochs)
        if data_test is not None:                               # If the test data is provided
            error_test = np.zeros(epochs)                       # Initialize the array to store the error during testing (epochs)
            out_test = np.zeros(data_test[1].shape)             # Initialize the array to store the output during testing
            
        a = []                                                  # Create a list of arrays of activations
        for l in self.layers:
            a.append(np.zeros(l))                               # One array of zeros per layer
            
        for k in range(epochs):                                 # Iterate through the epochs
            error_it = np.zeros(X.shape[0])                     # Initialize an array to store the errors during training (n examples)
            for it in range(X.shape[0]):                        # Iterate through the examples in the training set
                i = np.random.randint(X.shape[0])               # Select one random example
                a[0] = X[i]                                     # The activation of the first layer is the input values of the example

                                                                # Feed-forward
                for l in range(len(self.weights)):              # Iterate and compute the activation of each layer
                    a[l+1] = self.activation(np.dot(a[l], self.weights[l])) # Apply the activation function to the product input.weights
                
                error = a[-1] - y[i]                            # Compute the error: output - target
                error_it[it] = np.mean(error ** 2)              # Store the error of this iteration (average of all the outputs)
                deltas = [error * self.activation_deriv(a[-1])] # Ponderate the error by the derivative = delta
                
                                                                # Back-propagation
                                                                # We need to begin at the layer previous to the last one (out->in)
                for l in range(len(a) - 2, 0, -1):              # Append a delta for each layer
                    deltas.append(deltas[-1].dot(self.weights[l].T) * self.activation_deriv(a[l]))
                deltas.reverse()                                # Reverse the list (in->out)

                                                                # Update
                for i in range(len(self.weights)):              # Iterate through the layers
                    layer = np.atleast_2d(a[i])                 # Activation
                    delta = np.atleast_2d(deltas[i])            # Delta
                                                                # Compute the weight change using the delta f this layer
                                                                # and the change computed for the previous example for this layer
                    self.delta_weights[i] = (-learning_rate * layer.T.dot(delta)) + (momentum * self.delta_weights[i])
                    self.weights[i] += self.delta_weights[i]    # Update the weights
                
            error_train[k] = np.mean(error_it)                  # Compute the average of the error of all the examples
            if data_test is not None:                           # If a testing dataset was provided
                error_test[k], _ = self.compute_MSE(data_test)  # Compute the testing error after iteration k
            
        if data_test is None:                                   # If only a training data was provided
            return error_train                                  # Return the error during training
        else:
            return (error_train, error_test)                    # Otherwise, return both training and testing error
        
    def predict(self, x):
        '''
        Evaluates the network for a single observation
        '''
        x = np.array(x)
        temp = np.ones(x.shape[0]+1)
        temp[0:-1] = x
        a = temp
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a
    
    def compute_output(self, data):
        '''
        Evaluates the network for a dataset with multiple observations
        '''
        assert len(data.shape) == 2, 'data must be a 2-dimensional array'

        out = np.zeros((data.shape[0], self.n_outputs))
        for r in np.arange(data.shape[0]):
            out[r,:] = self.predict(data[r,:])
        return out
    
    def compute_MSE(self, data_test):
        '''
        Evaluates the network for a given dataset and
        computes the error between the target data provided
        and the output of the network
        '''
        assert len(data_test[0].shape) == 2, 'data[0] must be a 2-dimensional array'

        out = self.compute_output(data_test[0])
        return (np.mean((data_test[1] - out) ** 2), out)
