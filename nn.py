from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD
from keras.callbacks import Callback
import numpy as np


def neural_net(hiddenLayerDims, LoadWeights=''):
    # create nural network (keras model)
    neuralNet = Sequential()

    # create the dense input layer
    neuralNet.add(Dense(hiddenLayerDims[0], input_shape=(4,), input_dim=4))
    neuralNet.add(Activation('sigmoid'))

    # create second layer (first hidden layer)
    neuralNet.add(Dense(hiddenLayerDims[1]))
    neuralNet.add(Activation('sigmoid'))

    # create third and last layer
    neuralNet.add(Dense(4))
    neuralNet.add(Activation('softmax'))

    if LoadWeights:
        neuralNet.load_weights(LoadWeights)

    # create the optimizer (Stochastic Gradient Descent)
    sgd = SGD(lr=0.01, decay=0.0, momentum=0.0, nesterov=False)
    # Use mean squared error loss and SGD as optimizer
    neuralNet.compile(loss='mse', optimizer=sgd)

    return neuralNet
