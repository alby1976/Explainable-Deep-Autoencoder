import sys
from itertools import islice
from pathlib import Path
from typing import Tuple, Union, Iterable, Dict, Any, List, Optional

import numpy as np
import pandas as pd
import shap
import torch
import wandb
from matplotlib import pyplot as plt
from numpy import ndarray
from pandas import DataFrame, Series
from pyensembl import EnsemblRelease
from scipy.stats import levene, anderson, ks_2samp
from torch import device, Tensor
from torch.utils.data import DataLoader
from wandb import Image


class DataNormalization:
    def __init__(self, column_mask: Optional[ndarray] = None, column_names: Optional[ndarray] = None):
        from sklearn.preprocessing import MinMaxScaler

        super().__init__()
        self.scaler = MinMaxScaler()
        self.med_fold_change = None
        self.column_mask: Optional[ndarray] = column_mask
        self.column_names = column_names

    def fit(self, x, fold: bool):
        # the x is log2 transformed and then change to fold change relative to the row's median
        # Those columns whose column modian fold change relative to median is > 0 is keep
        # This module uses MaxABsScaler to scale the x

        # find column mask is not defined
        if self.column_mask is None:
            self.column_mask = np.median(x, axis=0) > 1

        # apply column mask
        print(f"x: {x.shape} column_mask {self.column_mask.shape}")
        print(f"x[:, self.column_mask]: {x[:, self.column_mask]}")
        tmp, median = get_transformed_data(x[:, self.column_mask], fold=fold)
        tmp, _ = get_fold_change(tmp, median=median, fold=fold)
        if self.column_names is not None:
            self.column_names = self.column_names[self.column_mask]

        print(f'\ntmp: {tmp.shape} mask: {self.column_mask.shape}', file=sys.stderr)
        # fit the x

        self.scaler = self.scaler.fit(X=tmp)

    def transform(self, x: Any, fold: bool) -> Union[Any, DataFrame]:
        # calculate fold change relative to the median after applying column mask
        print(f"x: {x.shape} column_mask: {self.column_mask.shape}")
        tmp, median = get_transformed_data(x[:, self.column_mask], fold=fold)
        tmp, _ = get_fold_change(tmp, median=median, fold=fold)
        print(f'\ntmp: {tmp.shape} mask: {self.column_mask.shape}', file=sys.stderr)
        # print(f'\ntmp: {tmp.shape} median: {median.shape}', file=sys.stderr)

        # Using MinMaxScaler() transform the x to be between (0..1)
        if self.column_names is None:
            return self.scaler.transform(X=tmp)
        else:
            return DataFrame(self.scaler.transform(X=tmp), columns=self.column_names)

    def save_column_mask(self, file: Path, column_name=None):
        data = self.column_mask[np.newaxis, :]
        df = pd.DataFrame(data=data, columns=column_name)
        df.to_csv(file)


# common functions
def get_device() -> device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_xy(dataloader: DataLoader) -> Tuple[Tensor, Tensor]:
    result_x = torch.cat([x for x, _ in dataloader], dim=0)
    result_y = torch.cat([y for _, y in dataloader], dim=0)
    return result_x, result_y


def get_test(self) -> Tuple[Tensor, Tensor]:
    result_x = torch.cat([x for x, _ in self.test_dataloader()], dim=0)
    result_y = torch.cat([y for _, y in self.test_dataloader()], dim=0)
    return result_x, result_y


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
def get_data(data: Path, index_col: Any = 0, header: Optional[str] = "infer") -> DataFrame:
    if not data.is_file():
        print(f'{data} does not exists.')
        sys.exit(-1)

    return pd.read_csv(data, index_col=index_col, header=header)


# returns the data file
def convert_gene_id_to_name(geno_id: DataFrame, col_name: ndarray) -> DataFrame:
    geno_id.rename(columns=dict(zip(geno_id.columns, col_name)), inplace=True)
    return geno_id


# returns the data that have been filtered allow with phenotypes
def get_data_phen(data: Path, filter_str: str, path_to_save_qc: Path) -> Tuple[DataFrame, Optional[Series]]:
    geno = get_data(data)
    geno = filter_data(geno, filter_str)
    create_dir(path_to_save_qc.parent)
    geno.to_csv(path_to_save_qc)

    return get_phen(geno)


def get_phen(geno: DataFrame) -> Tuple[DataFrame, Optional[Series]]:
    phen = None
    try:
        phen = geno.phen
        geno.drop(columns='phen', inplace=True)
    except KeyError:
        pass

    return geno, phen


def get_transformed_data(data, fold=False, median=None, col_names=None) -> \
        Tuple[Union[ndarray, DataFrame], Optional[ndarray]]:
    # filter out outliers

    # log2(TPM+0.25) transformation (0.25 to prevent negative inf)
    modified = np.log2(data + 0.25)
    med_exp: Optional[ndarray] = None

    if fold:
        modified, med_exp = get_fold_change(modified, median, fold)

    if col_names is not None:
        modified = DataFrame(data=modified, columns=col_names)

    return modified, med_exp


def get_fold_change(x, median, fold: bool) -> Tuple[ndarray, ndarray]:
    med_exp = np.median(x, axis=1) if median is None else median

    # fold change respect to  row median if fold is true
    result = np.asarray([x[i, :] - med_exp[i] for i in range(x.shape[0])]) if fold else x
    return result, med_exp


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


def create_gene_model(model_name: str, gene_model: Path, shap_values, gene_names: ndarray, sample_num: int,
                      top_num: int, node: int):
    # **generate gene model
    shap_values_mean = np.sum(abs(shap_values), axis=0) / sample_num
    shap_values_ln = np.log(shap_values_mean)  # *calculate ln^|shap_values_mean|
    gene_module: Union[ndarray, DataFrame] = np.stack((gene_names, shap_values_ln), axis=0)
    gene_module = gene_module.T
    gene_module = gene_module[np.argsort(gene_module[:, 1])]
    gene_module = gene_module[::-1]  # [starting index: stopping index: stepcount]
    gene_module = pd.DataFrame(gene_module)
    gene_module = gene_module.head(top_num)
    masking: Union[ndarray, bool] = gene_module[[1]] != -np.inf
    gene_module = gene_module[masking.all(axis=1)]
    if len(gene_module.index) > (1 / 4) * top_num:
        filename = f"{model_name}-shap({node:02}).csv"
        print(f'Creating {gene_model.joinpath(filename)} ...')
        gene_module.to_csv(gene_model.joinpath(filename), header=True, index=False, sep='\t')
        tbl = wandb.Table(dataframe=gene_module)
        tmp = f"{model_name}-shap({node:02})"
        wandb.log({tmp: tbl})
        print(f"...{gene_model.joinpath(filename)} Done ...\n")


def plot_shap_values(model_name: str, node: int, values, x_test: Union[ndarray, DataFrame, List], names: ndarray,
                     plot_type: str, plot_size, save_shap: Path):
    filename = f"{node:02}-{model_name}-{plot_type}.png"
    print(f"Creating {save_shap.joinpath(filename)} ...")
    shap.summary_plot(values, x_test, names, plot_type=plot_type, plot_size=plot_size, show=False)
    plt.savefig(f"{save_shap.joinpath(filename)}", dpi=100, format='png')
    plt.close()
    tmp = f"{node:02}-{model_name}-{plot_type}"
    image: Image = wandb.Image(str(save_shap.joinpath(filename)), caption="Top 20 features based on SHAP_values")
    wandb.log({tmp: image})
    print(f"{filename} Done")


def process_shap_values(save_bar: Path, save_scatter: Path, gene_model: Path, model_name: str, x_test, shap_values,
                        gene_names, sample_num, top_num, node):
    print(f"save_bar: {save_bar}\nsave_scatter: {save_scatter}\ngene_model: {gene_model}\nmodel_name: {model_name}\n"
          f"x_test:\n{x_test}\nshap_values:\n{shap_values}\ngene_names:{gene_names}\n")
    # save shap_gene_model
    create_gene_model(model_name, gene_model, shap_values, gene_names, sample_num, top_num, node)

    # generate bar char
    plot_shap_values(model_name, node, shap_values, x_test, gene_names, "bar", (15, 10), save_bar)
    # generate scatter chart
    plot_shap_values(model_name, node, shap_values, x_test, gene_names, "dot", (15, 10), save_scatter)
