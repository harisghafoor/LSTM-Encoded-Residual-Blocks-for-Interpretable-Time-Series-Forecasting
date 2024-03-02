"""
This module contains the implementation of the training process for LSTM-Encoded Residual Blocks for Interpretable Time Series Forecasting.

The module includes the following classes and functions:
- MetricsCallback: A callback class for computing evaluation metrics during training.
- Trainer: A class for training multiple models with different hyperparameters.

"""

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
from utils.config import *
from utils.dataloader import ElectricityLoader
from utils.loss import *


class MetricsCallback(tf.keras.callbacks.Callback):
    """
    Callback class to compute metrics on the test dataset after each epoch.

    Args:
        train_dataset (object): The training dataset object.
        test_dataset (object): The test dataset object.
        hyperparams (object): The hyperparameters object.

    Attributes:
        input_data (numpy.ndarray): The input data for the model.
        target (numpy.ndarray): The target data for the model.

    Methods:
        on_train_begin: Method called at the beginning of training.
        on_epoch_end: Method called at the end of each epoch.

    """

    def __init__(self, train_dataset, test_dataset, hyperparams):
        super().__init__()
        self.input_data = train_dataset.get_sequential_batch(
            win_len=hyperparams.horizon * hyperparams.history_lookback
        )
        self.target = np.array(test_dataset.ts_raw)

    def on_train_begin(self, logs={}):
        """
        Method called at the beginning of training.

        Args:
            logs (dict): Dictionary containing the training logs.

        """
        pass

    def on_epoch_end(self, epoch, logs={}):
        """
        Method called at the end of each epoch.

        Args:
            epoch (int): The current epoch number.
            logs (dict): Dictionary containing the training logs.

        """
        prediction_test = self.model.predict({"history": self.input_data["history"]})
        logs["smape_test"] = smape(preds=prediction_test["target"], labels=self.target)
        logs["mape_test"] = mape(preds=prediction_test["target"], labels=self.target)


class Trainer:
    """
    A class that represents a trainer for training and fitting models.

    Args:
        hyperparams (Parameters): The hyperparameters for the trainer.
        logdir (str): The directory path for logging.

    Attributes:
        hyperparams (list): A list of hyperparameters.
        history (list): A list to store the training history.
        forecasts (list): A list to store the forecasts.
        models (list): A list to store the models.
        logdir (str): The directory path for logging.
        folder_names (list): A list of folder names.

    Methods:
        generator: A generator function for generating batches of data.
        save_forecast: Saves the forecast to a file.
        fit: Fits the models using the training dataset.

    """

    def __init__(self, hyperparams: Parameters, logdir: str):
        """
        Initializes a new instance of the Trainer class.

        Args:
            hyperparams (Parameters): The hyperparameters for the trainer.
            logdir (str): The directory path for logging.

        """
        inp = dict(hyperparams._asdict())
        values = [v if isinstance(v, list) else [v] for v in inp.values()]
        self.hyperparams = [
            Parameters(**dict(zip(inp.keys(), v))) for v in product(*values)
        ]
        inp_lists = {k: v for k, v in inp.items() if isinstance(v, list)}
        values = [v for v in inp_lists.values()]
        variable_values = [dict(zip(inp_lists.keys(), v)) for v in product(*values)]
        folder_names = []
        for d in variable_values:
            folder_names.append(
                ";".join(["%s=%s" % (key, value) for (key, value) in d.items()])
            )
        self.history = []
        self.forecasts = []
        self.models = []
        self.logdir = logdir
        self.folder_names = folder_names
        for i, h in enumerate(self.hyperparams):
            self.models.append(
                NBeats(
                    hyperparams=h,
                    name=f"nbeats_model_{i}",
                    logdir=os.path.join(self.logdir, folder_names[i]),
                )
            )

    def generator(self, ds, hyperparams: Parameters):
        """
        A generator function for generating batches of data.

        Args:
            ds: The dataset.
            hyperparams (Parameters): The hyperparameters.

        Yields:
            dict: A dictionary containing the input data.
            dict: A dictionary containing the target data.

        """
        while True:
            batch = ds.get_batch(
                batch_size=hyperparams.batch_size,
                win_len=hyperparams.horizon * hyperparams.history_lookback,
                horizon=hyperparams.horizon,
                ts_sampling=hyperparams.ts_sampling,
            )

            yield {"history": batch["history"]}, {"target": batch["target"]}

    def save_forecast(self, forecast: np.ndarray, filename: str = "forecast.npy"):
        """
        Saves the forecast to a file.

        Args:
            forecast (np.ndarray): The forecast data.
            filename (str): The filename to save the forecast.

        """
        np.save(f"{PROJECT_ROOT}/{filename}", forecast)

    def fit(self, train_dataset, test_dataset, verbose=1):
        """
        Fits the models using the training dataset.

        Args:
            train_dataset: The training dataset.
            test_dataset: The test dataset.
            verbose (int): The verbosity level.

        """
        for i, hyperparams in enumerate(self.hyperparams):
            if verbose > 0:
                print(
                    f"Fitting model {i+1} out of {len(self.hyperparams)}, {self.folder_names[i]}"
                )

            path = f"results/{MODEL_VERSION}_split{self.models[i].hyperparams.split}/"
            pathlib.Path(f"{PROJECT_ROOT}/{path}").mkdir(parents=True, exist_ok=True)
            filename = os.path.join(path, self.folder_names[i] + ".npy")
            if os.path.exists(f"{PROJECT_ROOT}/{filename}"):
                continue

            boundary_step = hyperparams.epochs // 10
            boundary_start = (
                hyperparams.epochs - boundary_step * hyperparams.decay_steps - 1
            )

            boundaries = list(range(boundary_start, hyperparams.epochs, boundary_step))
            values = list(
                hyperparams.init_learning_rate
                * hyperparams.decay_rate ** np.arange(0, len(boundaries) + 1)
            )
            scheduler = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                boundaries=boundaries, values=values
            )

            lr = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=0)

            metrics = MetricsCallback(
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                hyperparams=hyperparams,
            )
            # tb = tf.keras.callbacks.TensorBoard(log_dir=self.models[i].logdir, embeddings_freq=10)

            if hyperparams.loss == "smape":
                loss = smape
            elif hyperparams.loss == "mape":
                loss = mape
            elif hyperparams.loss == "mae":
                loss = mae
            elif hyperparams.loss == "pmape":
                loss = get_pmape_loss(hyperparams.pinball_tau)
            elif hyperparams.loss == "pinball":
                loss = get_pinball_loss(hyperparams.pinball_tau)

            self.models[i].model.compile(
                optimizer=tf.keras.optimizers.Adam(), loss=loss
            )

            fit_output = self.models[i].model.fit(
                self.generator(ds=train_dataset, hyperparams=hyperparams),
                callbacks=[lr, metrics],  # tb
                epochs=hyperparams.epochs,
                steps_per_epoch=hyperparams.steps_per_epoch,
                verbose=verbose,
            )
            self.history.append(fit_output.history)

            model_forecast = self.models[i].forecast(train_dataset)
            self.save_forecast(forecast=model_forecast, filename=filename)


if __name__ == "__main__":
    train_dataset = ElectricityLoader(path="data", split="train")
    test_dataset = ElectricityLoader(path="data", split="test")
    batch = train_dataset.get_batch(win_len=12 * 2)
    # plt.plot(batch["history"][0])
    # plt.plot(np.concatenate([np.nan*np.zeros((24,)), batch["target"][0]]))
    trainer = Trainer(hyperparams=hyperparams, logdir=LOGDIR)
    trainer.fit(train_dataset=train_dataset, test_dataset=test_dataset)
