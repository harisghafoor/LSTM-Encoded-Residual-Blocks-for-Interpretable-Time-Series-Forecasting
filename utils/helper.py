import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, NamedTuple, Union
from tqdm.notebook import tqdm
import logging
import matplotlib.pyplot as plt
import time
from itertools import product
import gc
import pathlib
from glob import glob

from dateutil.relativedelta import relativedelta
from utils.config import *
from utils.dataloader import ElectricityLoader
from utils.loss import *

def get_forecasts(path, filt):
    a = []
    for f in tqdm(glob(os.path.join(PROJECT_ROOT, path, filt))):
        df = np.load(f)
        a.append(df)
    return a


def get_ensemble(forecasts):
    return np.mean(np.stack(forecasts, axis=-1), axis=-1)


def get_metrics(preds, labels):
    metrics = {}
    metrics["smape"] = 100 * smape(preds=preds, labels=labels).numpy()
    metrics["mape"] = 100 * mape(preds=preds, labels=labels).numpy()
    pe = 100 * (preds - labels) / labels
    metrics["pe_mean"] = np.mean(pe)
    metrics["pe_median"] = np.median(pe)
    return metrics


def get_stats(
    samples=10, ensemble_size=64, test_dataset=None, config_filt=None, path=None
):
    files = glob(os.path.join(PROJECT_ROOT, path, config_filt))
    all_repeats = set(
        [int(f.split(os.sep)[-1].split(";")[0].split("=")[-1]) for f in files]
    )
    preds = np.array(get_forecasts(path=path, filt=config_filt + ".npy"))

    metric_samples = []
    ensemble_samples = []
    for s in range(samples):
        ensemble_repeats = np.random.choice(
            list(all_repeats), size=ensemble_size, replace=False
        )
        ensemble = preds[ensemble_repeats].mean(axis=0)
        metric_samples.append(get_metrics(preds=ensemble, labels=test_dataset.ts_raw))
        ensemble_samples.append(ensemble)

    return pd.DataFrame(metric_samples), ensemble_samples


def save_ensemble_files(path, ensembles):
    """
    Save ensemble files as CSV.

    Args:
        path (str): The path where the ensemble files will be saved.
        ensembles (list): A list of ensembles to be saved as CSV files.

    Returns:
        None
    """
    pathlib.Path(f"{PROJECT_ROOT}/{path}").mkdir(parents=True, exist_ok=True)
    for i, e in enumerate(ensembles):
        filename = os.path.join(path, f"{i}.csv")
        df = pd.DataFrame(
            data=e,
            columns=[f"V{i+1}" for i in range(12)],
            index=[f"P{i+1}" for i in range(35)],
        )
        df.to_csv(filename)