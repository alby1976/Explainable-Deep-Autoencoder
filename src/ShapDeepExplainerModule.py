# python system library
from typing import Optional, Tuple, Union, Sequence, Any

# 3rd party modules
import numpy as np
import pytorch_lightning as pl
import shap
import torch
import torchmetrics.functional
from numpy import ndarray
from pandas import Series
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from sklearn.model_selection import train_test_split
from torch import nn, Tensor
from torch.utils.data import DataLoader, Dataset

# custom modules
from src.AutoEncoderModule import AutoGenoShallow


def process_shap_values(trainer: pl.Trainer, model: AutoGenoShallow, x: ndarray, hidden: ndarray, labels: Series,
                        test_split: float = 0.2, num_workers: int = 8, random_state: int = 42, shuffle: bool = False,
                        batch_size: int = 64, pin_memory: bool = True, drop_last: bool = False, save_dir):
    model.decoder = nn.Identity()


    #
    # deep_explainer =
    pass
