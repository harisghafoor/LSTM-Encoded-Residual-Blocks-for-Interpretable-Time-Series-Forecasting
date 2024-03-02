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

from .config import *

from model import NBeats

def smape(labels, preds):
    weights = tf.stop_gradient(
        tf.math.divide_no_nan(2.0, (tf.abs(preds) + tf.abs(labels))))
    return tf.reduce_mean(tf.abs(preds - labels) * weights)

def mape(labels, preds):
    weights = tf.math.divide_no_nan(1.0, tf.abs(labels))
    return tf.reduce_mean(tf.abs(preds - labels) * weights)

def get_pmape_loss(tau):
    """
    Returns a function that calculates the Pinball Loss with a given tau value.

    Parameters:
    tau (float): The quantile level for the Pinball Loss calculation.

    Returns:
    pmape_loss (function): A function that calculates the Pinball Loss.

    """
    def pmape_loss(labels, preds):
        weights = tf.math.divide_no_nan(1.0, tf.abs(labels))
        pinball = tf.where(labels > preds,
                           x=tau*(labels - preds),
                           y=(1-tau)*(preds-labels))
        return tf.reduce_mean(pinball * weights)
    return pmape_loss

def get_pinball_loss(tau):
    def pinball_loss(labels, preds):
        pinball = tf.where(labels > preds,
                           x=tau*(labels - preds),
                           y=(1-tau)*(preds-labels))
        return tf.reduce_mean(pinball)
    return pinball_loss

def mae(labels, preds):
    return tf.reduce_mean(tf.abs(preds - labels))
