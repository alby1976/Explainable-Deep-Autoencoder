import os
from itertools import islice
from pathlib import Path
from typing import Tuple, Union, Iterable, Dict, Any, List, Mapping

import numpy as np
import torch
from numpy import ndarray
from pandas import DataFrame, Series
from scipy.stats import anderson_ksamp, levene, anderson, ks_2samp, epps_singleton_2samp
from sklearn.preprocessing import MinMaxScaler
from torch import device, Tensor
from torchmetrics import Metric

"""
@dispatch(ndarray, ndarray, object)
def r2_value(y_true: ndarray, y_pred: ndarray, axis: object = None) -> object:
    y_ave = y_true.mean(axis=axis)
    # sse = np.sum(np.power(y_pred - y_ave, 2), axis=axis)
    ssr = np.sum(np.power(y_true - y_pred, 2), axis=axis)
    sst = np.sum(np.power(y_true - y_ave, 2), axis=axis)
    '''
    print(f'y_true: {y_true.shape}')
    print(f'y_ave: {y_ave.shape}\n{y_ave}\nssr: {ssr.shape}\n{ssr}\nsst: {sst.shape}\n{sst}\nssr/sst:{ssr/sst}\n'
          f'1 - (ssr/sst):\n{1 - (ssr/sst)}\n1 - np.divide(ssr, sst):\n{1 - np.divide(ssr, sst)}')
    '''
    return 1 - np.divide(ssr, sst)
"""


class DataNormalization:
    from sklearn.preprocessing import MinMaxScaler

    def __init__(self, column_names=None):
        super().__init__()
        self.scaler = MinMaxScaler()
        self.column_names = column_names

    def fit(self, x_train: Any):
        self.scaler = self.scaler.fit(X=x_train)

    def fit_transform(self, x_train: Any):
        if self.column_names is None:
            return self.scaler.fit_transform(X=x_train)
        else:
            return DataFrame(self.scaler.fit_transform(X=x_train), columns=self.column_names)

    def transform(self, x: Any):
        if self.column_names is None:
            return self.scaler.transform(X=x)
        else:
            return DataFrame(self.scaler.transform(X=x), columns=self.column_names)


def get_data(geno: DataFrame, path_to_save_qc: Path) -> DataFrame:
    try:
        geno.drop(columns='phen', inplace=True)
    except KeyError:
        pass

    geno = get_transformed_data(geno)
    create_dir(path_to_save_qc.parent)
    geno.to_csv(path_to_save_qc)
    return geno


# get dictionary values in a Tensor for a particular key in a list of dictionary
def get_dict_values_1d(key: str, lists: List[Dict[str, Tensor]], dim: int = 0) -> Tensor:
    return torch.stack([item[key] for item in lists], dim=dim)


def get_dict_values_2d(key: str, lists: List[Dict[str, Tensor]], dim: int = 0) -> Tensor:
    return torch.cat([item[key] for item in lists], dim=dim)


def data_parametric(*samples) -> bool:
    # print(f'samples: {type(samples)}\n\n{samples}\n\n')
    if len(samples) > 1:
        result1, _, _ = same_distribution_test(*samples)
        result2, _, _ = normality_test(samples[0])
        result3, _, _ = equality_of_variance_test(*samples)
    else:
        pass # TODO need to define

    return result1 and result2 and result3


def same_distribution_test(*samples) -> Tuple[bool, float, float]:
    stat: float
    crit: Union[ndarray, Iterable, int, float]

    stat, p_value = ks_2samp(samples[0], samples[1])

    if p_value < 0.05:
        return False, stat, p_value
    else:
        return True, stat, p_value


def normality_test(data: ndarray) -> Tuple[bool, float, float]:
    stat: float
    crit: Iterable

    stat, crit, _ = anderson(x=data, dist='norm')
    tmp = next(islice(crit, 2, 3))
    if tmp < stat:
        return False, stat, tmp
    else:
        return True, stat, tmp


def equality_of_variance_test(*samples: Tuple[ndarray, ...]) -> Tuple[bool, float, float]:
    stat: float
    p_value: float

    stat, p_value = levene(*samples, center='mean')
    if p_value < 0.5:
        return False, stat, p_value
    else:
        return True, stat, p_value


def get_transformed_data(data: DataFrame):
    # log2(TPM+0.25) transformation (0.25 to prevent negative inf)
    modified = DataFrame(data=np.log2(data + 0.25), columns=data.columns)

    med_exp = np.median(modified.values[:, 1:], axis=1)
    for i in range(modified.shape[0]):
        modified.iloc[i, 1:] = modified.values[i, 1:] - med_exp[i]  # fold change respect to median

    return modified


def create_dir(directory: Path):
    """make a directory (directory) if it doesn't exist"""
    directory.mkdir(parents=True, exist_ok=True)


def get_device() -> device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def filter_data(data: DataFrame, filter_str: str):
    try:
        return data[data.phen != filter_str]
    except AttributeError:
        return data
