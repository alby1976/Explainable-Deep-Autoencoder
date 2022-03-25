import sys
from itertools import islice
from pathlib import Path
from typing import Tuple, Union, Iterable, Dict, Any, List, Optional

import numpy as np
import torch
from numpy import ndarray
from pandas import DataFrame, Series
from scipy.stats import levene, anderson, ks_2samp
from torch import device, Tensor


class DataNormalization:
    def __init__(self):
        from sklearn.preprocessing import MinMaxScaler

        super().__init__()
        self.scaler = MinMaxScaler()
        self.med_fold_change = None
        self.column_mask: ndarray = np.asarray([])
        self.column_names = None

    def fit(self, x_train, column_names: Union[ndarray, Any] = None):
        # the data is log2 transformed and then change to fold change relative to the row's median
        # Those columns whose column modian fold change relative to median is > 0 is keep
        # This module uses MaxABsScaler to scale the data

        tmp: Union[ndarray, None] = None
        median: Union[ndarray, None] = None
        # find column mask
        self.column_mask: ndarray = np.median(x_train, axis=0) > 1

        # apply column mask
        tmp, median = get_transformed_data(x_train, fold=False)
        # tmp, _ = get_fold_change(tmp[:, self.column_mask], median=median)
        if column_names is not None:
            self.column_names = column_names[self.column_mask]

        print(f'\ntmp: {tmp.shape} mask: {self.column_mask.shape}', file=sys.stderr)
        # fit the data

        self.scaler = self.scaler.fit(X=tmp)

    def transform(self, x: Any):
        # calculate fold change relative to the median after applying column mask
        tmp, _ = get_transformed_data(x[:, self.column_mask])
        # tmp, median = get_transformed_data(x, fold=True)
        # tmp, _ = get_fold_change(tmp[:, self.column_mask], median=median)
        print(f'\ntmp: {tmp.shape} mask: {self.column_mask.shape}', file=sys.stderr)
        # print(f'\ntmp: {tmp.shape} median: {median.shape}', file=sys.stderr)

        # Using MinMaxScaler() transform the data to be between (0..1)
        if self.column_names is None:
            return self.scaler.transform(X=tmp)
        else:
            return DataFrame(self.scaler.transform(X=tmp), columns=self.column_names)


# common functions
def get_device() -> device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# get dictionary values in a Tensor for a particular key in a list of dictionary
def get_dict_values_1d(key: str, lists: List[Dict[str, Tensor]], dim: int = 0) -> Tensor:
    return torch.stack([item[key] for item in lists], dim=dim)


def get_dict_values_2d(key: str, lists: List[Dict[str, Tensor]], dim: int = 0) -> Tensor:
    return torch.cat([item[key] for item in lists], dim=dim)


def data_parametric(*samples) -> bool:
    # print(f'samples: {type(samples)}\n\n{samples}\n\n')
    result1: bool = False
    result2: bool = False
    result3: bool = False
    if len(samples) > 1:
        result1, _, _ = same_distribution_test(*samples)
        result2, _, _ = normality_test(samples[0])
        result3, _, _ = equality_of_variance_test(*samples)
    else:
        pass  # TODO need to define

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


def create_dir(directory: Path):
    """make a directory (directory) if it doesn't exist"""
    directory.mkdir(parents=True, exist_ok=True)


def get_data(geno: DataFrame, path_to_save_qc: Path, filter_str: str) -> Tuple[DataFrame, Series]:
    geno = filter_data(geno, filter_str)
    phen = None
    try:
        phen = geno.phen
        geno.drop(columns='phen', inplace=True)
    except KeyError:
        pass

    create_dir(path_to_save_qc.parent)
    geno.to_csv(path_to_save_qc)
    return geno, phen


def get_transformed_data(data, fold=False, median=None, col_names=None) -> Tuple[Union[ndarray, DataFrame], ndarray]:
    # filter out outliers

    # log2(TPM+0.25) transformation (0.25 to prevent negative inf)
    modified = np.log2(data + 0.25)
    med_exp: ndarray = np.asarray([])

    if fold:
        modified, med_exp = get_fold_change(modified, median)

    if col_names is not None:
        return DataFrame(data=modified, columns=col_names), med_exp

    return modified, med_exp


def get_fold_change(x, median) -> Tuple[ndarray, ndarray]:
    med_exp = np.median(x, axis=1) if median is None else median
    # fold change respect to  row median
    return np.asarray([x[i, :] - med_exp[i] for i in range(x.shape[0])]), med_exp


def filter_data(data: DataFrame, filter_str: str):
    try:
        return data[data.phen.isin(filter_str)]
    except AttributeError:
        return data


def med_var(data, axis=0):
    med = np.median(data, axis=axis)
    tmp = np.median(np.power(data - med, 2), axis=axis)
    return tmp


def float_or_none(value: str) -> Optional[float]:
    """
    float_or_none.

    Examples:
        >>> import argparse
        >>> parser = argparse.ArgumentParser()
        >>> _ = parser.add_argument('--foo', type=float_or_none)
        >>> parser.parse_args(['--foo', '4.5'])
        Namespace(foo=4.5)
        >>> parser.parse_args(['--foo', 'none'])
        Namespace(foo=None)
        >>> parser.parse_args(['--foo', 'null'])
        Namespace(foo=None)
        >>> parser.parse_args(['--foo', 'nil'])
        Namespace(foo=None)
    """
    if value.strip().lower() in ("none", "null", "nil"):
        return None
    return float(value)
