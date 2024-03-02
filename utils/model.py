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

from .config import Parameters

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
class NBeats:
    """
    NBeats class represents the N-Beats model for time series forecasting.

    Args:
        hyperparams (Parameters): The hyperparameters for the N-Beats model.
        name (str, optional): The name of the model. Defaults to 'NBeats'.
        logdir (str, optional): The directory for logging. Defaults to 'logs'.
        num_nodes (int, optional): The number of nodes in the model. Defaults to 100.
    """

    def __init__(self, hyperparams: Parameters, name: str='NBeats', logdir: str='logs', num_nodes: int = 100):
        super(NBeats, self).__init__()
        self.hyperparams = hyperparams
        self.name=name
        self.logdir=logdir
        self.num_nodes = num_nodes
        self.input_size = self.hyperparams.history_lookback*self.hyperparams.horizon

        self.nbeats_layers = []
        self.nbeats_layers.append(NBeatsBlockWithLSTM(hyperparams=hyperparams,
                                              input_size=self.input_size,
                                              output_size=hyperparams.horizon,
                                              lstm_units=hyperparams.lstm_units,
                                              name=f"nbeats_{0}")
                                 )
        for i in range(1, hyperparams.num_blocks):
            if self.hyperparams.block_sharing:
                self.nbeats_layers.append(self.nbeats_layers[0])
            else:
                self.nbeats_layers.append(NBeatsBlockWithLSTM(hyperparams=hyperparams,
                                                      input_size=self.input_size,
                                                      output_size=hyperparams.horizon,
                                                      lstm_units=hyperparams.lstm_units,
                                                      name=f"nbeats_{i}")
                                         )

        inputs, outputs = self.get_model()
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
        self.inputs = inputs
        self.outputs = outputs
        self.model = model

    def get_model(self):
        """
        Get the N-Beats model architecture.

        Returns:
            tuple: A tuple containing the inputs and outputs of the model.
        """
        history_in = tf.keras.layers.Input(shape=(self.hyperparams.history_lookback*self.hyperparams.horizon,), name='history')

        level = tf.reduce_max(history_in, axis=-1, keepdims=True)
        history_delevel = tf.math.divide_no_nan(history_in, level)

        backcast, forecast = self.nbeats_layers[0](inputs=history_delevel)
        for nb in self.nbeats_layers[1:]:
            backcast, forecast_layer = nb(inputs=backcast)
            forecast = forecast + forecast_layer

        forecast = forecast * level

        inputs = {'history': history_in}
        outputs = {'target': forecast}
        return inputs, outputs

    def forecast(self, train_dataset):
        """
        Generate forecasts using the N-Beats model.

        Args:
            train_dataset: The training dataset.

        Returns:
            numpy.ndarray: The forecasted values.
        """
        input_data = train_dataset.get_sequential_batch(win_len=self.hyperparams.horizon*self.hyperparams.history_lookback)
        return self.model.predict({"history": input_data["history"]})['target']

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
