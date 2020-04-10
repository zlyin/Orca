## import packages
import os
import sys
import numpy as np

class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
        # layers is a list of n_neurons, eg [2, 2, 1]
        self.layers = layers
        self.alpha = alpha
        
        self.W = []
        for i in np.arange(0, len(layers) - 2):
            # bias trick
            weights = np.random.randn(self.layers[i] + 1, self.layers[i + 1] + 1) 
            self.W.append(weights / np.sqrt(self.layers[i]))
        
        # initiate last hidden layer & output layer
        weights = np.random.randn(self.layers[-2] + 1, self.layers[-1])
        self.W.append(weights / np.sqrt(self.layers[-2]))
        pass


    # define a python magic method, __repr__
    def __repr__(self):
        structures = [str(l) for l in self.layers]
        return "NeuralNetwork {}".format("-".join(structures))

    # define activation function
    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def sigmoid_derivative(self, ac_output):
        # ac_output is activation output of each layer
        return ac_output * (1 - ac_output)

    # define fit method
    def fit(self, X, y, epochs=100, displayUpdate=100):
        # insert bias into X
        X = np.c_[X, np.ones(X.shape[0])]

        # store displayed loss to plot learning curve
        displayLoss = []
        
        # loop over epoches
        for epoch in np.arange(0, epochs):
            # loop over each data print
            for (x, target) in zip (X, y):
                # feedforward & BP
                self.fit_partial(x, target)

            # print out status periodically
            if epoch == 0 or (epoch + 1) % displayUpdate == 0:
                loss = self.calculate_loss(X, y)
                displayLoss.append(loss)
                print("[INFO] epoch = %d, loss = %.7f" % (epoch, loss))
        return displayLoss


    # define feedforward - BP method
    def fit_partial(self, x, gt):
        # initiate layer_outputs, first layer is input x
        Layer_outputs = [np.atleast_2d(x)]  # ensure x in shape(1, n_features)

        ## Feedforward
        for layer in np.arange(0,len(self.W)):
            # calculate layer output
            net = Layer_outputs[layer].dot(self.W[layer])

            # activation function
            out = self.sigmoid(net)

            # store layeroutput of current layer
            Layer_outputs.append(out)

        ## Backpropagation
        
        # compute error at first
        error = Layer_outputs[-1] - gt

        # initiate deltas of layers
        Deltas = [error * self.sigmoid_derivative(Layer_outputs[-1])]

        # loop over layers in revers/BP, starting from last hidden layer
        for layer in np.arange(len(Layer_outputs) - 2, 0, -1):
            # layer i -> j, calculate delta(i) by delta(j)
            delta = Deltas[-1].dot(self.W[layer].T) 
            # derivative of activation function
            delta = delta * self.sigmoid_derivative(Layer_outputs[layer])
            # store delta of current layer
            Deltas.append(delta)

        ## reverse deltas
        Deltas = Deltas[::-1]

        # update weights
        for layer in np.arange(0, len(self.W)):
            # ensure to match dimensions
            #print(Layer_outputs[layer].shape, Deltas[layer].shape)
            self.W[layer] += -self.alpha * Layer_outputs[layer].T.dot(Deltas[layer])
        pass


    def predict(self, X, addBias=True):
        # ensure X is in shape(n_samples, n_fetures) 
        p = np.atleast_2d(X)

        # add bias
        if addBias:
            p = np.c_[p, np.ones(p.shape[0])]

        # loop over layers to predict/feedforward process
        for layer in np.arange(0, len(self.W)):
            p = self.sigmoid(np.dot(p, self.W[layer]))

        # return
        return p


    def calculate_loss(self, X, targets):
        # ensures targets are in shape (n_samples, 1)
        targets = np.atleast_2d(targets)
        # predict
        preds = self.predict(X, addBias=False) # Because self.fit() already add bias into X! 
        loss = 0.5 * np.sum((preds - targets) ** 2) 
        return loss







    
