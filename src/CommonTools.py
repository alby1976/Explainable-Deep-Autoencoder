from itertools import islice
from pathlib import Path
from typing import Tuple, Union, Iterable

import torch
from numpy import ndarray
from pandas import DataFrame
from scipy.optimize import anderson
from scipy.stats import anderson_ksamp, levene
from torch import device


def data_parametric(*samples: ndarray) -> bool:
    result1, _, _ = same_distribution_test(samples)
    result2, _, _ = normality_test(samples[0])
    result3, _, _ = equality_of_variance_test(samples)
    return result1 and result2 and result3


def same_distribution_test(*samples: Tuple[ndarray, ...]) -> Tuple[bool, float, float]:
    stat: float
    crit: Union[ndarray, Iterable, int, float]

    stat, crit, _ = anderson_ksamp(samples=samples)

    if crit[2] < stat:
        return False, stat, crit[2]
    else:
        return True, stat, crit[2]


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

    stat, p_value = levene(samples, center='mean')
    if p_value < 0.5:
        return False, stat, p_value
    else:
        return True, stat, p_value


def get_normalized_data(data: DataFrame) -> DataFrame:
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    return DataFrame(data=scaler.fit_transform(data), columns=data.columns)


def create_dir(directory: Path):
    """make a directory (directory) if it doesn't exist"""
    directory.mkdir(parents=True, exist_ok=True)


def get_device() -> device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
