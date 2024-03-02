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

class ElectricityLoader:
    """
    A class for loading electricity time series data.

    Parameters:
    - path (str): The path to the data directory. Default is "data/".
    - split (str): The split of the data to load. Default is 'train'.

    Methods:
    - get_val_split(split_horizon=1): Returns a tuple of train and test datasets split based on the given horizon.
    - get_batch(batch_size=64, win_len=14*2, horizon=12, ts_sampling='uniform'): Returns a batch of training data.
    - get_sequential_batch(win_len=14*2): Returns a batch of sequential training data.
    """

    def __init__(self, path="data/", split='train'):
        self.path = path
        self.split = split
        ts_raw = pd.read_csv(os.path.join(path,f'Electricity-{split}.csv')).iloc[:,1:].values.astype(np.float32)
        self.ts_raw = []

        for ts in ts_raw:
            self.ts_raw.append(ts[~np.isnan(ts)])

        self.ts_weight = np.zeros((len(self.ts_raw),), dtype=np.float32)
        for i,ts in enumerate(self.ts_raw):
            self.ts_weight[i] = len(ts)
        self.ts_weight = self.ts_weight / self.ts_weight.sum()

    def get_val_split(self, split_horizon=1):
        """
        Returns a tuple of train and test datasets split based on the given horizon.

        Parameters:
        - split_horizon (int): The number of time steps to split the data. Default is 1.

        Returns:
        - train_dataset (ElectricityLoader): The train dataset.
        - test_dataset (ElectricityLoader): The test dataset.
        """
        train_dataset = ElectricityLoader(path=self.path, split='train')
        test_dataset = ElectricityLoader(path=self.path, split='train')
        ts_raw = []
        for ts in train_dataset.ts_raw:
            ts_raw.append(ts[:-split_horizon*12])
        train_dataset.ts_raw = ts_raw
        ts_raw = []
        for ts in test_dataset.ts_raw:
            ts_raw.append(ts[-split_horizon*12:])
        test_dataset.ts_raw = ts_raw
        return train_dataset, test_dataset

    def get_batch(self, batch_size=64, win_len=14*2, horizon=12, ts_sampling='uniform'):
        """
        Returns a batch of training data.

        Parameters:
        - batch_size (int): The size of the batch. Default is 64.
        - win_len (int): The length of the input window. Default is 14*2.
        - horizon (int): The number of time steps to predict. Default is 12.
        - ts_sampling (str): The method for sampling time series. Default is 'uniform'.

        Returns:
        - batch (dict): A dictionary containing the input history and target values.
        """
        target_ts = np.zeros((batch_size, horizon), dtype=np.float32)
        history_ts = np.zeros((batch_size, win_len), dtype=np.float32)

        if ts_sampling == "uniform":
            ts_idxs = np.random.choice(np.arange(len(self.ts_raw)), size=batch_size, replace=True)
        elif ts_sampling == "ts_weight":
            ts_idxs = np.random.choice(np.arange(len(self.ts_raw)), size=batch_size, replace=True, p=self.ts_weight)

        for i, ts_id in enumerate(ts_idxs):
            ts = self.ts_raw[ts_id]
            sampling_point = np.random.choice(np.arange(win_len, len(ts)-horizon+1), size=1, replace=False)[0]
            history_ts[i,:] = ts[sampling_point-win_len:sampling_point]
            target_ts[i,:] = ts[sampling_point:sampling_point+horizon]

        batch = {"history": history_ts, "target": target_ts}
        return batch

    def get_sequential_batch(self, win_len=14*2):
        """
        Returns a batch of sequential training data.

        Parameters:
        - win_len (int): The length of the input window. Default is 14*2.

        Returns:
        - batch (dict): A dictionary containing the input history.
        """
        history_ts = np.zeros((len(self.ts_raw), win_len), dtype=np.float32)
        for i, ts in enumerate(self.ts_raw):
            history_ts[i,:] = ts[-win_len:]
        return {"history": history_ts}


