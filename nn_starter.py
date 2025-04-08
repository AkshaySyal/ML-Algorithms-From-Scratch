import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

class NN:
    """
        Implementation of a 2-layer feed-forward network with softmax output.
    """
    def __init__(self, n_hidden, n_output, epochs, batch_size, learning_rate):
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        """
        Compute the sigmoid function for the input here.
        """
        ### YOUR CODE HERE

        ### END YOUR CODE
        return s

    def sigmoid_deriv(self, x):
        """
        Compute the derivative of the sigmoid function here.
        """
        ### YOUR CODE HERE

        ### END YOUR CODE
        return d

    def softmax(self, x):
        """
        Compute softmax function for input.
        """
        ### YOUR CODE HERE

        ### END YOUR CODE
        return s

    def feed_forward(self, X):
        """
        Forward propagation
        return cache: a dictionary containing the activations of all the units
               output: the predictions of the network
        """
        ### YOUR CODE HERE

        ### END YOUR CODE
        cache = {}
        cache['Z1'] = Z1
        cache['A1'] = A1
        cache['Z2'] = Z2
        cache['A2'] = A2
        return cache, output

    def back_propagate(self, X, y, cache):
        """
        Return the gradients of the parameters
        """
        ### YOUR CODE HERE

        ### END YOUR CODE

        grads = {}
        grads['W1'] = dW1
        grads['b1'] = db1
        grads['W2'] = dW2
        grads['b2'] = db2

        return grads

    def init_weights(self, n_input):
        ### YOUR CODE HERE

        ### END YOUR CODE

    def update_weights(self, grads):
        ### YOUR CODE HERE

        ### END YOUR CODE

    def compute_loss(self, y, output):
        """
        Return the cross-entropy loss
        """
        ### YOUR CODE HERE

        ### END YOUR CODE
        return loss

    def train(self, X_train, y_train, X_val, y_val):
        (n, m) = X_train.shape
        self.init_weights(m)

        ### YOUR CODE HERE

        ### END YOUR CODE

    def test(self, X_test, y_test):
        cache, output = self.feed_forward(X_test)
        accuracy = self.compute_accuracy(output, y_test)
        return accuracy

    def compute_accuracy(self, y, output):
        accuracy = (np.argmax(y, axis=1) == np.argmax(output, axis=1)).sum() * 1. / y.shape[0]
        return accuracy

    def one_hot_labels(self, y):
        one_hot_labels = np.zeros((y.size, self.n_output))
        one_hot_labels[np.arange(y.size), y.astype(int)] = 1
        return one_hot_labels

def main():
    nn = NN(n_hidden=300, n_output=10, epochs=30, batch_size=1000, learning_rate=5)
    np.random.seed(100)

    X, y = fetch_openml('mnist_784', return_X_y=True, as_frame=False)
    X = (X / 255).astype('float32')

    X_train, y_train = X[0:60000], y[0:60000]
    y_train = nn.one_hot_labels(y_train)
    p = np.random.permutation(60000)
    X_train = X_train[p]
    y_train = y_train[p]

    X_val = X_train[0:10000]
    y_val = y_train[0:10000]
    X_train = X_train[10000:]
    y_train = y_train[10000:]

    X_test, y_test = X[60000:], y[60000:]
    y_test = nn.one_hot_labels(y_test)

    nn.train(X_train, y_train, X_val, y_val)

    accuracy = nn.test(X_test, y_test)
    print(f'Test accuracy: {accuracy}')

if __name__ == '__main__':
    main()
