import pdb
import numpy as np
import mnist_loader
from sklearn.preprocessing import scale
import time
import mnist_loader

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
        pdb.set_trace()
        return np.sum((1 - y) * np.log(1 - y_) + y * np.log(y_)) / len(y_)

    def softmax(self, y):
        # Here we need to take softmax of prediction and y
        return np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)

    def rms_loss(self, y_, y):
        return np.sqrt(np.sum(np.mean((y_ - y) ** 2)))

    def sigmoid(self, y):
        ans = 1 / (1 + np.exp(-y))

        return ans

    def calc_num_grads(self, X, y):
        numericals = []
        num_grads_w = map(lambda w: np.zeros(w.shape), self.weights)

        for chk_idx in range(len(self.weights)):
            for i in range(len(self.weights[chk_idx])):
                for j in range(len(self.weights[chk_idx][i])):
                    copied_weights = map(lambda w: np.copy(w), self.weights)

                    copied_weights[chk_idx][i][j] -= 0.001
                    prediction_1 = self.predict(X, copied_weights, self.biases)
                    copied_weights[chk_idx][i][j] += 0.002
                    prediction_2 = self.predict(X, copied_weights, self.biases)
                    loss_1 = self.rms_loss(prediction_1, y)
                    loss_2 = self.rms_loss(prediction_2, y)
                    num_grad = (loss_2 - loss_1) / 0.002
                    num_grads_w[chk_idx][i][j] = num_grad

        num_grads_b = map(lambda b: np.zeros(b.shape), self.biases)
        for chk_idx in range(len(self.biases)):
            for i in range(len(self.biases[chk_idx])):
                copied_biases = map(lambda b: np.copy(b), self.biases)

                copied_biases[chk_idx][i] -= 0.001
                prediction_1 = self.predict(X, self.weights, copied_biases)
                copied_biases[chk_idx][i] += 0.002
                prediction_2 = self.predict(X, self.weights, copied_biases)
                loss_1 = self.rms_loss(prediction_1, y)
                loss_2 = self.rms_loss(prediction_2, y)
                num_grad = (loss_2 - loss_1) / 0.002
                num_grads_b[chk_idx][i] = num_grad

        return (num_grads_w, num_grads_b)

    def sigmoid_derivative(self, y):
        return self.sigmoid(y) * (1 - self.sigmoid(y))

    def fit(self, X, y, epochs=1000000, batch_size=1000, rate=1, gradient_check=False):
        for e in range(epochs):
            for i in range(len(X) / batch_size):
                idxs = np.random.choice(np.arange(len(X)), batch_size)
                batch_x = X[idxs]
                batch_y = y[idxs]
                self.backprop(batch_x, batch_y)

            scores = self.predict(X)
            #print(scores)
            #print(y)
            print('epoch: ', e, 'loss: ', self.rms_loss(scores, y))
            print('epoch: ', e, 'accuracy: ', self.accuracy(scores, y))

            # Don't use backprop, update via num gradients instead
            #(scores, all_gradients, dWs) = self.backprop(X, y, rate)
            #self.backprop(X, y)
            #if gradient_check:
            #    self.check_gradients(X, y, all_gradients, dWs)
            #if i % 100 == 0:
            #    scores = self.predict(X)
            #    #print(scores)
            #    #print(y)
            #    print('loss: ', self.rms_loss(scores, y))
            #    print('accuracy: ', self.accuracy(scores, y))

    def predict(self, x, weights=None, biases=None):
        if not weights:
            weights = self.weights
        if not biases:
            biases = self.biases

        activation = x
        for weight, bias in zip(weights, biases):
            z = activation.dot(weight) + bias
            activation = self.sigmoid(z)
        return activation

    def rmse(self, y_, y):
        return np.sqrt(np.mean(np.sum((y - y_)**2)))

    def set_gradient(self, y_, y):
        return -2 / (len(y)) * (y - y_)

    def accuracy(self, y_, y):
        prediction_idxs = np.argmax(y_, axis=1)
        return y[range(len(y)), prediction_idxs].mean()

    def backprop(self, X, y, rate=1e-3):
        # Forward pass
        zs = []
        activations = [X]
        activation = X
        for weight, bias in zip(self.weights, self.biases):
            z = activation.dot(weight) + bias
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)

        # Change the name to make this more consistent
        #scores = self.softmax(activation)
        scores = activation
        gradients = self.set_gradient(scores, y)

        dWs = []
        dbs = []
        gradients = gradients * self.sigmoid_derivative(zs[-1])
        dWs.append(activations[-2].T.dot(gradients))
        dbs.append(np.sum(gradients, axis=0))
        all_gradients = [gradients]
        ## Back propagate
        for i in reversed(range(0, len(self.weights) - 1)):
            # Pass gradients back
            #pdb.set_trace()
            da_dz = self.weights[i + 1].T
            dz_da = self.sigmoid_derivative(zs[i])
            gradients = gradients.dot(da_dz) * dz_da

            all_gradients.append(da_dz)
            all_gradients.append(dz_da)
            dW = activations[i].T.dot(gradients)
            db = np.sum(gradients, axis=0)
            dWs.insert(0, dW)
            dbs.insert(0, db)

        #print(map(lambda x: x.shape, self.weights))
        #print(map(lambda x: x.shape, dWs))
        for i in range(len(self.weights)):
            self.weights[i] -= rate * dWs[i]
            self.biases[i] -= rate * dbs[i]
        return (scores, all_gradients, dWs)

# Test data
nn = NeuralNetwork([2, 10, 3])

def generate_random_data(n=10, x_dim=2, y_dim=3):
    X = np.zeros((n, 2))
    y = np.zeros((n, 3))
    for i in range(n):
        X[i] = np.random.rand(x_dim)
        idx = np.argmax(np.random.rand(y_dim))
        y[i][idx] = 1

    return X, y

#X, y = generate_random_data(20)
#X = scale(X)
##X = scale(np.array([
##    [36, 100000],
##    [30, 88000],
##    [27, 36000],
##    [32, 77000],
##    [32, 77000],
##]))
##
##y = np.array([
##    [1, 0, 0],
##    [1, 0, 0],
##    [0, 1, 0],
##    [0, 0, 1],
##    [0, 0, 1],
##])
#
#nn.fit(X, y, gradient_check=True)

# Mnist training data
nn = NeuralNetwork([784, 32, 10])

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
np.random.shuffle(training_data)

X = np.zeros((len(training_data), 784, 1))
y = np.zeros((len(training_data), 10, 1))

for i in range(len(training_data)):
    x, yp = training_data[i]
    X[i, :] = x
    y[i, :] = yp
print(X.shape)
print(y.shape)
X = X.reshape(X.shape[0], X.shape[1])
y = y.reshape(y.shape[0], y.shape[1])
nn.fit(X, y)
