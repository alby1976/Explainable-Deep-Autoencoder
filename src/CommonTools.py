import sys
from itertools import islice
from pathlib import Path
from typing import Tuple, Union, Iterable, Dict, Any, List, Optional

import numpy as np
import pandas as pd
import torch
from multipledispatch import dispatch
from numpy import ndarray
from pandas import DataFrame, Series
from pyensembl import EnsemblRelease
from scipy.stats import levene, anderson, ks_2samp
from torch import device, Tensor


class DataNormalization:
    def __init__(self, column_mask=None):
        from sklearn.preprocessing import MinMaxScaler

        super().__init__()
        self.scaler = MinMaxScaler()
        self.med_fold_change = None
        self.column_mask: Union[ndarray, None] = column_mask
        self.column_names = None

    def fit(self, x, fold: bool, column_names: Union[ndarray, None] = None):
        # the x is log2 transformed and then change to fold change relative to the row's median
        # Those columns whose column modian fold change relative to median is > 0 is keep
        # This module uses MaxABsScaler to scale the x

        tmp: Union[ndarray, None] = None
        median: Union[ndarray, None] = None
        # find column mask is not defined
        if self.column_mask is None:
            self.column_mask = np.median(x, axis=0) > 1

        # apply column mask
        tmp, _ = get_transformed_data(x[:, self.column_mask], fold=fold)
        # tmp, _ = get_fold_change(tmp[:, self.column_mask], median=median)
        if column_names is not None:
            self.column_names = column_names[self.column_mask]

        print(f'\ntmp: {tmp.shape} mask: {self.column_mask.shape}', file=sys.stderr)
        # fit the x

        self.scaler = self.scaler.fit(X=tmp)

    def transform(self, x: Any, fold: bool):
        # calculate fold change relative to the median after applying column mask
        tmp, _ = get_transformed_data(x[:, self.column_mask], fold=fold)
        # tmp, median = get_transformed_data(x, fold=True)
        # tmp, _ = get_fold_change(tmp[:, self.column_mask], median=median)
        print(f'\ntmp: {tmp.shape} mask: {self.column_mask.shape}', file=sys.stderr)
        # print(f'\ntmp: {tmp.shape} median: {median.shape}', file=sys.stderr)

        # Using MinMaxScaler() transform the x to be between (0..1)
        if self.column_names is None:
            return self.scaler.transform(X=tmp)
        else:
            return DataFrame(self.scaler.transform(X=tmp), columns=self.column_names)

    def save_column_mask(self, file: Path, column_name=None, version: int = 104):
        gene_names = get_gene_names(ensembl_release=version, gene_list=column_name)
        data = self.column_mask[np.newaxis, :]
        df = pd.DataFrame(data=data, columns=gene_names)
        df.to_csv(file)


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


# returns the data file
@dispatch(Path)
def get_data(data: Path) -> DataFrame:
    if not data.is_file():
        print(f'{data} does not exists.')
        sys.exit(-1)

    return pd.read_csv(data, index_col=0)


# returns the data file along with col_mask
@dispatch(Path, Path)
def get_data(data: Path, col_mask_file: Path) -> Tuple[DataFrame, ndarray]:
    geno: DataFrame
    col_mask = get_data(col_mask_file)

    if data is None:
        pass
    else:
        geno = get_data(data)

    return geno, col_mask.to_numpy()


# returns the data that have been filtered allow with phenotypes
@dispatch(Path, Path, str)
def get_data(data: Path, path_to_save_qc: Path, filter_str: str) -> Tuple[DataFrame, Series]:
    geno = pd.read_csv(data, index_col=0)
    geno = filter_data(geno, filter_str)
    create_dir(path_to_save_qc.parent)
    geno.to_csv(path_to_save_qc)

    return get_phen(geno)


def get_phen(geno: DataFrame) -> Tuple[DataFrame, Union[Series, None]]:
    phen = None
    try:
        phen = geno.phen
        geno.drop(columns='phen', inplace=True)
    except KeyError:
        pass

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


def filter_data(data: DataFrame, filter_str):
    if filter_str is not None:
        try:
            return data[data.phen.isin(filter_str)]
        except AttributeError:
            pass
    return data


def med_var(data, axis=0):
    med = np.median(data, axis=axis)
    tmp = np.median(np.power(data - med, 2), axis=axis)
    return tmp


def get_gene_ids(ensembl_release: int, gene_list: np.ndarray) -> np.ndarray:
    gene_data = EnsemblRelease(release=ensembl_release, species='human', server='ftp://ftp.ensembl.org/')
    gene_data.download()
    gene_data.index()
    ids = []
    for gene in gene_list:
        try:
            ids.append((gene_data.gene_ids_of_gene_name(gene_name=gene)[0]).replace('\'', ''))
        except ValueError:
            ids.append(gene)
    return np.array(ids)


def get_gene_names(ensembl_release: int, gene_list: np.ndarray) -> np.ndarray:
    gene_data = EnsemblRelease(release=ensembl_release, species='human', server='ftp://ftp.ensembl.org/')
    names = []
    tmp: str

    for gene in gene_list:
        try:
            tmp = gene_data.gene_name_of_gene_id(gene).replace('\'', '')
        except ValueError:
            tmp = ""

        if len(tmp) > 0:
            names.append(tmp)
        else:
            names.append(gene)
    return np.array(names)


def float_or_none(value: str) -> Optional[float]:
    if value.strip().lower() in ("none", "null", "nil"):
        return None
    return float(value)
