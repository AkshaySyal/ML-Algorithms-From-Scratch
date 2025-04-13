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
        s = 1 / (1 + np.exp(-x))
        ### END YOUR CODE
        return s

    def sigmoid_deriv(self, x):
        """
        Compute the derivative of the sigmoid function here.
        """
        ### YOUR CODE HERE
        d = self.sigmoid(x)*(1-self.sigmoid(x))
        ### END YOUR CODE
        return d

    def softmax(self, x):
        """
        Compute softmax function for input.
        """
        ### YOUR CODE HERE
        d = np.sum(np.exp(x),axis=1,keepdims=True) #[[1,2,3],[3,4,5]]
        s = np.exp(x)/d
        ### END YOUR CODE
        return s

    def feed_forward(self, X):
        """
        Forward propagation
        return cache: a dictionary containing the activations of all the units
               output: the predictions of the network
        """
        ### YOUR CODE HERE
        with open('weights_1.pkl', 'rb') as f:
            weights_1 = pickle.load(f)
        with open('weights_2.pkl', 'rb') as f:
            weights_2 = pickle.load(f)
        with open('bias_1.pkl', 'rb') as f:
            bias_1 = pickle.load(f)
        with open('bias_2.pkl', 'rb') as f:
            bias_2 = pickle.load(f)
        
        A1 = X @ weights_1 + bias_1 # 1000,300 # Pre-activation
        Z1 = self.sigmoid(A1) # Hidden layer activation 
        A2 = Z1 @ weights_2 + bias_2 # 1000,10 # Output layer pre-activation
        Z2 = self.softmax(A2) # Output layer activation

        ### END YOUR CODE
        cache = {}
        cache['Z1'] = Z1
        cache['A1'] = A1
        cache['Z2'] = Z2
        cache['A2'] = A2
        return cache, A2

    def back_propagate(self, X, y, cache):
        """
        Return the gradients of the parameters
        """
        ### YOUR CODE HERE
        with open('weights_2.pkl', 'rb') as f:
            weights_2 = pickle.load(f)
  
        dA2 = cache['Z2'] - y # 1000,10
        dW2 = cache['Z1'].T @ dA2 / self.batch_size  # 300,10
        db2 = np.sum(dA2, keepdims=True, axis=0) / self.batch_size # 1,10 

        dZ1 = dA2 @ weights_2.T # 1000,300
        dA1 = dZ1 * self.sigmoid_deriv(cache['Z1']) # 1000,300
        dW1 = X.T @ dA1 / self.batch_size # 784,300
        db1 = np.sum(dA1,keepdims=True,axis=0) / self.batch_size # 1,300

        ### END YOUR CODE
        grads = {}
        grads['W1'] = dW1
        grads['b1'] = db1
        grads['W2'] = dW2
        grads['b2'] = db2

        return grads

    def init_weights(self, n_input):
        ### YOUR CODE HERE
        weights_1 = np.random.normal(loc=0, scale=1, size=(784,300))
        with open('weights_1.pkl', 'wb') as f:
            pickle.dump(weights_1, f)

        # 300x10
        weights_2 = np.random.normal(loc=0, scale=1, size=(300,10))
        with open('weights_2.pkl', 'wb') as f:
            pickle.dump(weights_2, f)

        # Bias terms (1x300), initialized to 0
        bias_1 = np.zeros((1,300))
        with open('bias_1.pkl', 'wb') as f:
            pickle.dump(bias_1, f)

        bias_2 = np.zeros((1,10))
        with open('bias_2.pkl', 'wb') as f:
            pickle.dump(bias_2, f)

        ### END YOUR CODE

    def update_weights(self, grads):
        ### YOUR CODE HERE
        with open('weights_1.pkl','rb') as f:
            W1 = pickle.load(f)
        with open('bias_1.pkl','rb') as f:
            b1 = pickle.load(f)
        with open('weights_2.pkl','rb') as f:
            W2 = pickle.load(f)
        with open('bias_2.pkl','rb') as f:
            b2 = pickle.load(f)

        W1 -= self.learning_rate * grads['W1']
        b1 -= self.learning_rate * grads['b1']
        W2 -= self.learning_rate * grads['W2']
        b2 -= self.learning_rate * grads['b2']

        with open('weights_1.pkl','wb') as f:
            pickle.dump(W1, f)
        with open('bias_1.pkl','wb') as f:
            pickle.dump(b1, f)
        with open('weights_2.pkl','wb') as f:
            pickle.dump(W2, f)
        with open('bias_2.pkl','wb') as f:
            pickle.dump(b2, f)
        ### END YOUR CODE

    def compute_loss(self, y, output):
        """
        Return the cross-entropy loss
        """
        ### YOUR CODE HERE
        # y: (1000,10) one-hot encoded vector
        # output: 1000,10 softmax values
        eps = 1e-12
        output_clipped = np.clip(output, eps, 1. - eps)
        loss_per_sample = -np.sum(y * np.log(output_clipped),axis=1)
        loss = np.mean(loss_per_sample)

        ### END YOUR CODE
        return loss

    def train(self, X_train, y_train, X_val, y_val):
        # Shapes
        # X_Train: 50000x784
        # y_train: 50000,1
        # X_val: 10000x784
        # y_val: 10000x1

        (n, m) = X_train.shape 
        self.init_weights(m)
        num_batches = n // self.batch_size # 50000/1000 = 50
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        
        ### YOUR CODE HERE
        for epoch in range(self.epochs):
            for i in range(num_batches):
                start_idx = i * self.batch_size
                end_idx = start_idx + self.batch_size

                X_batch = X_train[start_idx:end_idx]
                y_batch = y_train[start_idx:end_idx]

                # Forward pass
                cache,output = self.feedforward(X_batch)

                # Backprop 
                grads = self.back_propagate(X_batch, y_batch, cache)
                self.update_weights(grads)
            
            # Loss and Accuracy calculation on entire training set
            train_cache, train_output = self.feed_forward(X_train)
            train_loss = self.compute_loss(y_train, train_output)
            train_acc = self.compute_accuracy(train_output, y_train)

            # Forward pass on entire validation set
            val_cache, val_output = self.feed_forward(X_val)
            val_loss = self.compute_loss(y_val, val_output)
            val_acc = self.compute_accuracy(val_output, y_val)

            # Store for plotting
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            print(f"Epoch {epoch+1}/{self.epochs} "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        
        # Loss curves
        plt.figure()
        plt.plot(range(1, self.epochs+1), train_losses, label='Train Loss')
        plt.plot(range(1, self.epochs+1), val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.show()

        # Accuracy curves
        plt.figure()
        plt.plot(range(1, self.epochs+1), train_accs, label='Train Accuracy')
        plt.plot(range(1, self.epochs+1), val_accs, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.show()

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
