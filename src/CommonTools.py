import os
from itertools import islice
from pathlib import Path
from typing import Tuple, Union, Iterable, Dict, Any, List, Mapping

import numpy as np
import torch
from numpy import ndarray
from pandas import DataFrame, Series
from scipy.stats import anderson_ksamp, levene, anderson, ks_2samp, epps_singleton_2samp
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


def r2_value(y_true: Tensor, y_pred: Tensor, dim: int = 0) -> object:
    y_ave = torch.mean(y_true, dim=dim)
    # sse = torch.sum(torch.pow(y_pred - y_ave, 2), dim=dim)
    ssr = torch.sum(torch.pow(y_true - y_pred, 2), dim=dim)
    sst = torch.sum(torch.pow(y_true - y_ave, 2), dim=dim)
    '''
    print(f'y_true: {y_true.size()}')
    print(f'y_ave: {y_ave.size()}\n{y_ave}\nssr: {ssr.size()}\n{ssr}\nsst: {sst.size()}\n{sst}\nssr/sst:{ssr/sst}\n'
          f'1 - (ssr/sst):\n{1 - (ssr / sst)}\n')
    '''
    return 1 - (ssr / sst)


def r2_value_weighted(y_true: Tensor, y_pred: Tensor, dim: int = 0) -> Union[Metric, Tensor, int, float,
                                                                             Mapping[str, Union[Metric, Tensor, int,
                                                                                                float]]]:
    y_ave = torch.mean(y_true, dim=dim)
    sst = torch.sum(torch.pow(y_true - y_ave, 2), dim=dim)
    sst_sum = torch.sum(sst)
    raw = r2_value(y_true=y_true, y_pred=y_pred, dim=dim)
    # print(f'r2_value_weighted: {torch.nansum(sst / sst_sum * raw)}')
    return torch.sum(sst / sst_sum * raw)


def get_data(geno: DataFrame, path_to_save_qc: Path) -> DataFrame:
    return get_normalized_data(get_filtered_data(geno, path_to_save_qc))


def get_filtered_data(geno: DataFrame, path_to_save_qc: Path) -> DataFrame:
    try:
        geno.drop(columns='phen', inplace=True)
    except KeyError:
        pass
    geno_var: Union[Series, int] = geno.var()
    geno_var = geno_var[geno_var < 1]
    tmp = geno_var.index.values
    geno.drop(tmp, axis=1, inplace=True)
    create_dir(path_to_save_qc.parent)
    geno.to_csv(path_to_save_qc)
    return geno


# merge list to single dict
def merge_list_dict(lists) -> Dict[Any, Any]:
    result = {}
    for tmp in lists:
        result = {**result, **tmp}

    return result


def save_tensor(x: Tensor, file: Path):
    with open(file, "w") as fh:
        for i in range(x.size(0)):
            row: list = []
            for j in range(x.size(1)):
                # do something
                row.append(x[i][j])
                row.append(',')
            list[-1] = "\n"
            fh.writelines(list)
        fh.flush()
        os.fsync()
        fh.close()


# get dictionary values in a Tensor for a particular key in a list of dictionary
def get_dict_values_1d(key: str, lists: List[Dict[str, Tensor]], dim: int = 0) -> Tensor:
    return torch.stack([item[key] for item in lists], dim=dim)


def get_dict_values_2d(key: str, lists: List[Dict[str, Tensor]], dim: int = 0) -> Tensor:
    return torch.cat([item[key] for item in lists], dim=dim)


def data_parametric(*samples) -> bool:
    # print(f'samples: {type(samples)}\n\n{samples}\n\n')
    result1, _, _ = same_distribution_test(*samples)
    result2, _, _ = normality_test(samples[0])
    result3, _, _ = equality_of_variance_test(*samples)
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


def get_normalized_data(data: DataFrame) -> DataFrame:
    from sklearn.preprocessing import MinMaxScaler

    # log2(TPM+0.25) transformation (0.25 to prevent negative inf)
    modified = DataFrame(data=np.log2(data + 0.25), columns=data.columns)

    med_exp = np.median(modified.values[:, 1:], axis=1)
    for i in range(modified.shape[0]):
        modified.iloc[i, 1:] = modified.values[i, 1:] - med_exp[i]  # fold change respect to median

    scaler = MinMaxScaler()
    return DataFrame(data=scaler.fit_transform(modified), columns=modified.columns)


def create_dir(directory: Path):
    """make a directory (directory) if it doesn't exist"""
    directory.mkdir(parents=True, exist_ok=True)


def get_device() -> device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
