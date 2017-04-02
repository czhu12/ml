import pdb
import numpy as np
import mnist_loader
from sklearn.preprocessing import scale

class NeuralNetwork():
    def __init__(self, layers):
        self.layers = layers
        self.weights = []
        for i in range(len(layers) - 1):
            weight = np.random.normal(0.1, 0.1, (layers[i], layers[i + 1]))
            self.weights.append(weight)

        self.biases = []
        for i in range(1, len(layers)):
            bias = np.random.normal(0.1, 0.1, layers[i])
            self.biases.append(bias)

    def cross_entropy_loss(self, y_, y):
        return np.sum((1 - y) * np.log(1 - y_) + y * np.log(y_))

    def softmax(self, y):
        # Here we need to take softmax of prediction and y
        return np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)

    def sigmoid(self, y):
        return 1 / (1 + np.exp(-y))

    def sigmoid_derivative(self, y):
        return self.sigmoid(y) * (1 - self.sigmoid(y))

    def fit(self, X, y, epochs=10000):
        for i in range(epochs):
            scores = self.backprop(X, y, i)
            if (i % 1000) == 0:
                print("prediction: ")
                print(scores)
                print("actual: ")
                print(y)

    def set_gradient(self, y_, y):
        return y_ - y

    def backprop(self, X, y, epoch_num, rate=1e-3):
        # Forward pass
        zs = []
        activations = [X]
        output = X
        for i in range(0, len(self.layers) - 1):
            weight = self.weights[i]
            bias = self.biases[i]
            z = output.dot(weight) + bias
            zs.append(z)
            output = self.sigmoid(z)
            activations.append(output)

        # Change the name to make this more consistent
        scores = self.softmax(output)
        loss = self.cross_entropy_loss(scores, y)
        gradients = self.set_gradient(scores, y)

        if (epoch_num % 1000) == 0:
            print("first gradient")
            print(gradients)

        dWs = []
        dbs = []
        # Back propagate
        for i in reversed(range(len(self.weights))):
            dW = activations[i].T.dot(gradients)
            db = np.sum(gradients, axis=0)
            dWs.insert(0, dW)
            dbs.insert(0, db)

            # Pass gradients back
            gradients = gradients.dot(self.weights[i].T) * self.sigmoid_derivative(activations[i])

        if (epoch_num % 1000) == 0:
            print("last gradient")
            print(gradients)
            pdb.set_trace()

        for i in range(len(self.weights)):
            self.weights[i] -= rate * dWs[i]
            self.biases[i] -= rate * dbs[i]

        return scores

    def predict(self, X):
        output = X
        for i in range(0, len(self.layers) - 1):
            weight = self.weights[i]
            bias = self.biases[i]
            output = output.dot(weight) + bias
            output = self.sigmoid(output)

        #output.dot(weight[-1]) + bias[-1]
        #return self.softmax(output)
        return output

nn = NeuralNetwork([3, 5, 2])
X = scale(np.array([
    [36, 100000, 180],
    [30, 88000, 150],
    [27, 36000, 160],
    [32, 77000, 140],
    [36, 100000, 180],
    [30, 88000, 150],
    [27, 36000, 160],
    [32, 77000, 140],
    [36, 100000, 180],
    [30, 88000, 150],
    [27, 36000, 160],
    [32, 77000, 140]
]))

y = np.array([
    [1, 0],
    [1, 0],
    [0, 1],
    [0, 1],
    [1, 0],
    [1, 0],
    [0, 1],
    [0, 1],
    [1, 0],
    [1, 0],
    [0, 1],
    [0, 1],
    ])

nn.fit(X, y)
x = scale(np.array([
    [28, 180000, 180],
    [35, 37000, 200],
    [32, 84000, 170],
    [28, 180000, 180],
    [35, 37000, 200],
    [32, 84000, 170],
    [28, 180000, 180],
    [35, 37000, 200],
    [32, 84000, 170],
]))
#print('prediction')
#print(nn.predict(x))
