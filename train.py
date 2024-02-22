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

train_dataset = ElectricityLoader(path='data', split='train')
test_dataset = ElectricityLoader(path='data', split='test')

batch = train_dataset.get_batch(win_len=12*2)

plt.plot(batch["history"][0])
plt.plot(np.concatenate([np.nan*np.zeros((24,)), batch["target"][0]]))