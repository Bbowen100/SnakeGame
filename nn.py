from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD, Adam
from keras.callbacks import Callback
import numpy as np


def neural_net(hiddenLayerDims, LoadWeights=''):
    # create nural network (keras model)
    neuralNet = Sequential()

    # create the dense input layer
    neuralNet.add(Dense(hiddenLayerDims[0], input_dim=4))
    neuralNet.add(Activation('relu'))

    # create second layer (first hidden layer)
    neuralNet.add(Dense(hiddenLayerDims[0]))
    neuralNet.add(Activation('relu'))


    # create third and last layer
    neuralNet.add(Dense(4))
    neuralNet.add(Activation('softmax'))

    if LoadWeights:
        neuralNet.load_weights(LoadWeights)

    # create the optimizer (Stochastic Gradient Descent)
    sgd = SGD(lr=0.3, decay=1e-2, momentum=0.9, nesterov=True)
    # Use mean squared error loss and SGD as optimizer
    neuralNet.compile(loss='mean_squared_error', optimizer=sgd ,
              metrics=['accuracy'])

    return neuralNet
