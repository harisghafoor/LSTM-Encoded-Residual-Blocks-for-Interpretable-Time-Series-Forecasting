import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, NamedTuple, Union
from tqdm.notebook import tqdm
import logging
from matplotlib import pyplot as plt
import time
from itertools import product
import gc
import pathlib
from glob import glob

from dateutil.relativedelta import relativedelta
class NBeatsBlockWithLSTM(tf.keras.layers.Layer):
    def __init__(self, hyperparams, input_size, output_size, **kwargs):
        super(NBeatsBlockWithLSTM, self).__init__(**kwargs)
        self.hyperparams = hyperparams
        self.input_size = input_size
        self.output_size = output_size
        self.lstm_layer = tf.keras.layers.LSTM(hyperparams['lstm_units'], return_sequences=True)
        self.flatten_layer = tf.keras.layers.Flatten()
        self.fc_layers = []
        for i in range(hyperparams['block_layers']):
            self.fc_layers.append(tf.keras.layers.Dense(hyperparams['hidden_units'],
                                                        activation=tf.nn.relu,
                                                        kernel_regularizer=tf.keras.regularizers.l2(hyperparams['weight_decay']),
                                                        name=f"fc_{i}"))
        self.backcast = tf.keras.layers.Dense(input_size, activation='relu', name="backcast")
        self.forecast = tf.keras.layers.Dense(output_size, activation=None, name="forecast")
    def call(self, inputs):
        x = tf.expand_dims(inputs, axis=2)  # Add extra dimension to input tensor
        x = self.lstm_layer(x)
        x = self.flatten_layer(x)
        for i in range(self.hyperparams['block_layers']):
            x = self.fc_layers[i](x)
        backcast = self.backcast(inputs)
        forecast = self.forecast(x)
        return backcast, forecast

if __name__ == 'main':
    # Define the input and output sizes
    input_size = 10
    output_size = 5
    #Define the hyperparameters for the block
    hyperparams = {
        'block_layers': 3,
        'hidden_units': 32,
        'weight_decay': 0.001,
        'lstm_units':3
    }

    # Create a sample input tensor
    inputs = tf.random.normal(shape=(1, input_size))

    # Create an instance of the NBeatsBlock layer
    nbeats_block = NBeatsBlockWithLSTM(hyperparams, input_size, output_size)

    # Call the layer to compute the backcast and forecast
    backcast, forecast = nbeats_block(inputs)
    print(backcast)
    # # Check the shape of the outputs
    # assert backcast.shape == (1, input_size)
    # assert forecast.shape == (1, output_size)

    # # Check that the backcast tensor has non-negative values
    # assert tf.reduce_any(backcast >= 0)

    # # Check that the forecast tensor has non-negative values
    # assert tf.reduce_any(forecast >= 0)

    # # Check that the forecast tensor is not all zero
    # assert tf.reduce_any(forecast != 0)
