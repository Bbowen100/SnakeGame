# SnakeGame
Using a Neural Network to train a snake to play the snake game. This project uses the Keras module for the construction of the Neural Netowork and PyGame to create the User Interface.

## Keras: Deep Learning library for TensorFlow and Theano

Keras is a high-level neural networks library, written in Python and capable of running on top of either [TensorFlow](https://github.com/tensorflow/tensorflow) or [Theano](https://github.com/Theano/Theano). It was developed with a focus on enabling fast experimentation. *Being able to go from idea to result with the least possible delay is key to doing good research.*

Use Keras if you need a deep learning library that:

- Allows for easy and fast prototyping (through total modularity, minimalism, and extensibility).
- Supports both convolutional networks and recurrent networks, as well as combinations of the two.
- Supports arbitrary connectivity schemes (including multi-input and multi-output training).
- Runs seamlessly on CPU and GPU.

Read the documentation at [Keras.io](http://keras.io).

Keras is compatible with: __Python 2.7-3.5__.

------------------

## Installation

Keras uses the following dependencies:

- numpy, scipy
- pyyaml
- HDF5 and h5py (optional, required if you use model saving/loading functions)
- Optional but recommended if you use CNNs: cuDNN.


*When using the TensorFlow backend:*

- TensorFlow
    - [See installation instructions](https://github.com/tensorflow/tensorflow#download-and-setup).

*When using the Theano backend:*

- Theano
    - [See installation instructions](http://deeplearning.net/software/theano/install.html#install).

To install Keras, `cd` to the Keras folder and run the install command:
```sh
sudo python setup.py install
```

You can also install Keras from PyPI:
```sh
sudo pip install keras
```

------------------


## Switching from TensorFlow to Theano

By default, Keras will use TensorFlow as its tensor manipulation library. [Follow these instructions](http://keras.io/backend/) to configure the Keras backend.

------------------

