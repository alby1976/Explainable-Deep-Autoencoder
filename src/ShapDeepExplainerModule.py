# python system library
# 3rd party modules
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Union, List, Tuple, Any

import numpy as np
import pandas as pd
import shap
import torch
import wandb
from matplotlib import pyplot as plt
from numpy import ndarray
from pandas import DataFrame
from torch import nn

from AutoEncoderModule import AutoGenoShallow


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
        print(f'{gene_model}({node}).csv')
        gene_module.to_csv(f'{gene_model}({node}).csv', header=True, index=False, sep='\t')
        tbl = wandb.Table(dataframe=gene_module)
        wandb.log({f"{model_name}({node})": tbl})


def plot_shap_values(model_name: str, node: int, values, x_test: Union[ndarray, DataFrame, List], plot_type: str,
                     plot_size, save_shap: Path):
    shap.summary_plot(values, x_test, plot_type=plot_type, plot_size=plot_size)
    print(f'{save_shap}-{plot_type}({node}).png')
    plt.savefig(f'{save_shap}-{plot_type}({node}).png', dpi=100, format='png')
    plt.close()
    tmp = f"{model_name}-{plot_type}({node})"
    wandb.log({tmp: wandb.Image(f"{save_shap}.png")})


def process_shap_values(save_bar: Path, save_scatter: Path, gene_model: Path, model_name: str, x_test, shap_values,
                        gene_names, sample_num, top_num, node):
    # save shap_gene_model
    create_gene_model(model_name, gene_model, shap_values, gene_names, sample_num, top_num, node)

    # generate bar char
    plot_shap_values(model_name, node, shap_values, x_test, "bar", (15, 10), save_bar)
    # generate scatter chart
    plot_shap_values(model_name, node, shap_values, x_test, "dot", (15, 10), save_scatter)


def create_shap_values(model: AutoGenoShallow, model_name: str, gene_model: Path, save_bar: Path, save_scatter: Path,
                       top_rate: float = 0.05):
    shap_values: Union[
        ndarray, List[ndarray], Tuple[List[Union[ndarray, List[ndarray]]], Any], List[Union[ndarray, List[ndarray]]]]

    # setup
    model.decoder = nn.Identity()
    batch = next(iter(model.train_dataloader()))
    genes, _ = batch
    x_train = genes[:100]

    batch = next(iter(model.val_dataloader()))
    x_test = torch.cat([batch[0] for batch in model.test_dataloader()])

    gene_names: ndarray = model.gene_names
    top_num: int = int(top_rate * len(gene_names))  # top_rate is the percentage of features to be calculated

    explainer = shap.DeepExplainer(model, x_train)
    shap_values = explainer.shap_values(x_test, top_num, "max")  # shap_values contains values for all nodes
    x_test = x_test.detach().cpu().numpy()
    with ThreadPoolExecutor(max_workers=8) as pool:
        params = ((save_bar, save_scatter, gene_model, model_name, x_test, shap_values, model.gene_names,
                   model.sample_size, top_num, node) for node in shap_values)
        pool.map(lambda x: process_shap_values(*x), params)
