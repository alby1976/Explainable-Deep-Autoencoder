# python system library
# 3rd party modules
from pathlib import Path
from typing import Union, List, Tuple, Any

import shap
import torch
import wandb
from matplotlib import pyplot as plt
from numpy import ndarray
from torch import nn

from src.AutoEncoderModule import AutoGenoShallow


def plot_shap_values(model_name, values, x_test, plot_type, plot_size, save_shap: Path):
    shap.summary_plot(values, x_test, plot_type=plot_type, plot_size=plot_size)
    print(f'{save_shap}.png')
    plt.savefig(f'{save_shap}.png', dpi=100, format='png')
    plt.close()
    tmp = f"{model_name}-{plot_type}"
    wandb.log({tmp: wandb.Image(f"{save_shap}.png")})


def process_shap_values(model: AutoGenoShallow, model_name, save_dir: Path, save_bar: Path, save_scatter: Path):
    shap_values: Union[
        ndarray, List[ndarray], Tuple[List[Union[ndarray, List[ndarray]]], Any], List[Union[ndarray, List[ndarray]]]]

    model.decoder = nn.Identity()
    x_train, _ = model.dataset.train_dataset
    x_train = torch.from_numpy(x_train)
    x_test, _ = model.dataset.val_dataset
    x_test = torch.from_numpy(x_test)

    explainer = shap.DeepExplainer(model, x_train)
    # TODO: need to determine out_rank if any
    shap_values = explainer.shap_values(x_test) # shap_values contains values for all nodes

    # TODO: need to manipulate data

    # generate bar char
    plot_shap_values(model_name, shap_values, x_test, "bar", (15, 10), save_bar)
    # generate scatter chart
    plot_shap_values(model_name, shap_values, x_test, "dot", (15, 10), save_scatter)

    #
    # deep_explainer =
