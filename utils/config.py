import os
PROJECT_ROOT = '/content/drive/My Drive/electricity'
os.makedirs('/content/drive/My Drive/electricity',exist_ok=True)
DATASETS_PATH = "data"
os.makedirs(os.path.join(PROJECT_ROOT,DATASETS_PATH))
LOGDIR = f"./logs"
MODEL_VERSION = 'TEST'
from typing import Dict, NamedTuple, Union
class Parameters(NamedTuple):
    split: int
    repeat: int
    epochs: int
    steps_per_epoch: int
    block_layers: int
    hidden_units: int
    num_blocks: int
    block_sharing: bool
    horizon: int
    history_lookback: int
    init_learning_rate: float
    decay_steps: int
    decay_rate: float
    loss: str
    pinball_tau: float
    batch_size: int
    weight_decay: float
    ts_sampling: str
    lstm_units:int

hyperparams_dict = {
    "split": 0, # 0 is test split, 1 is validation split
    "repeat": list(range(0, 2)),
    "epochs": 1,
    "steps_per_epoch": [1],
    "block_layers": 3,
    "hidden_units": 512,
    "num_blocks": 3,
    "block_sharing": True, # [True, False]
    "horizon": 12,
    "history_lookback": 1,
    "init_learning_rate": 1e-3,
    "decay_steps": 3,
    "decay_rate": 0.5,
    "loss": ["pinball", "pmape"], # ["pinball", "mape", "smape", "pmape"]
    "pinball_tau": [0.35], # This is selected to minimize the bias on the validation set
    "batch_size": 256,
    "weight_decay": 0,
    "ts_sampling": ["ts_weight"], # ["uniform", "ts_weight"]
    "lstm_units" : 3,
    }
HORIZON = 12
hyperparams = Parameters(**hyperparams_dict)

