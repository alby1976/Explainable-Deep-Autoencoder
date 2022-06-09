# Use Python to plot SHAP figure (include both bar chart and scatter chart) and generate gene module based on SHAP value
import argparse
import platform
from pathlib import Path
from typing import Any, Dict, Tuple, Union, List, Optional

import numpy as np
import pandas as pd
import shap
import sklearn
import wandb
import xgboost as xgb
from matplotlib import pyplot as plt
from numpy import ndarray
from pandas import DataFrame
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from wandb import Image, Table

from CommonTools import create_dir, get_phen, get_data, DataNormalization, convert_gene_id_to_name


def get_last_model(directory: Path):
    file_path: Path
    time: float
    time, file_path = max((f.stat().st_mtime, f) for f in directory.iterdir())
    return file_path


def predict_shap_values(boost, phen, unique, unique_count, gene, hidden_vars, test_split, shuffle, random_state,
                        num_workers, dm, fold, sample_num, ids, top_num, gene_model, model_name, save_bar,
                        save_scatter, column_num, i, summary_tbl: Table) -> Tuple[int, float]:
    print(f'**** Processing {i + 1} out of {column_num} columns ****')
    x_train: Any
    x_test: Any
    y_train: Any
    y_test: Any
    phen_train: Optional[ndarray]
    phen_test: Optional[ndarray]
    if phen is not None and unique.size > 1 and np.min(unique_count) > 1:
        x_train, x_test, y_train, y_test, phen_train, phen_test = train_test_split(gene.values,
                                                                                   hidden_vars.values,
                                                                                   phen,
                                                                                   test_size=test_split,
                                                                                   stratify=phen,
                                                                                   random_state=random_state)
    else:
        x_train, x_test, y_train, y_test, = train_test_split(gene.values,
                                                             hidden_vars.values,
                                                             test_size=test_split,
                                                             shuffle=shuffle,
                                                             random_state=random_state)

    my_model = xgb.XGBRegressor(booster="gbtree", max_depth=20, random_state=random_state,
                                eval_metric="rmse",
                                n_estimators=100, objective='reg:squarederror') if boost else \
        RandomForestRegressor(bootstrap=True, oob_score=False, max_depth=20,
                              random_state=random_state, n_estimators=100,
                              n_jobs=num_workers)

    print(f"\nx_train: {x_train.shape} y_train: {y_train.shape}\n dm:\n{dm.column_mask}")
    dm.fit(x_train, fold)
    my_model.fit(dm.transform(x_train, fold), y_train)
    y_pred = my_model.predict(dm.transform(x_test, fold))
    r2: float = sklearn.metrics.r2_score(y_true=y_test, y_pred=y_pred)
    explainer = shap.TreeExplainer(my_model)
    # **explainer = shap.KernelExplainer(my_model.predict, x = x_test.iloc[0:10])
    x_test = dm.transform(x_test, fold)
    shap_values = explainer.shap_values(x_test)
    # process shap values and generate gene model
    features, bar, scatter = process_shap_values(save_bar, save_scatter, gene_model, model_name, x_test, shap_values,
                                                 ids, sample_num, top_num, i)
    summary_tbl.add_data(i, wandb.Image(bar), wandb.Image(scatter), features, r2)
    return i, r2


def create_gene_model(model_name: str, gene_model: Path, shap_values, gene_names: ndarray, sample_num: int,
                      top_num: int, node: int) -> int:
    # **generate gene model
    shap_values_mean = np.sum(abs(shap_values), axis=0) / sample_num
    # *calculate ln^|shap_values_mean|

    shap_values_ln = np.log(shap_values_mean)
    gene_module: Union[ndarray, DataFrame] = np.stack((gene_names, shap_values_ln), axis=0)
    gene_module = gene_module.T
    gene_module = gene_module[np.argsort(gene_module[:, 1])]
    gene_module = gene_module[::-1]  # [starting index: stopping index: stepcount]

    convert_nan = np.vectorize(lambda x: np.nan if np.isneginf(np.log(x)) else np.log(x))
    gene_module = pd.DataFrame(gene_module)
    gene_module = gene_module.head(top_num)
    print(f"before gene_module:\n{gene_module}\n")

    gene_module = gene_module.dropna(subset=[1]).reset_index(drop=True)  # drop rows that contain a nan
    print(f"after gene_module:\n{gene_module}\n")
    # if len(gene_module.index) > 1/4 * top_num:
    filename = f"{model_name}-shap({node:02}).csv"
    print(f'Creating {gene_model.joinpath(filename)} ...')
    gene_module.to_csv(gene_model.joinpath(filename), header=True, index=False, sep='\t')
    tbl = wandb.Table(dataframe=gene_module)
    tmp = f"{model_name}-shap({node:02})"
    wandb.log({tmp: tbl})
    print(f"...{gene_model.joinpath(filename)} Done ...\n")
    return gene_module.shape[0]


def plot_shap_values(model_name: str, node: int, values, x_test: Union[ndarray, DataFrame, List], names: ndarray,
                     plot_type: str, plot_size, save_shap: Path):
    print(f"save_shap ")
    filename = f"{node:02}-{model_name}-{plot_type}.png"
    filename = save_shap.joinpath(filename)
    print(f"Creating {filename} ...")
    shap.summary_plot(values, x_test, names, plot_type=plot_type, plot_size=plot_size, show=False)
    plt.savefig(str(filename), dpi=100, format='png')
    plt.close()
    tmp = f"{node:02}-{model_name}-{plot_type}"
    image: Image = wandb.Image(str(filename), caption="Top 20 features based on SHAP_values")
    wandb.log({tmp: image})
    print(f"{filename} Done")
    return str(filename)


def process_shap_values(save_bar: Path, save_scatter: Path, gene_model: Path, model_name: str, x_test, shap_values,
                        gene_names, sample_num, top_num, node) -> Tuple[int, str, str]:
    print(f"save_bar: {save_bar}\nsave_scatter: {save_scatter}\ngene_model: {gene_model}\nmodel_name: {model_name}\n"
          f"ene_names:{gene_names.shape}\n")
    # save shap_gene_model
    num_features = create_gene_model(model_name, gene_model, shap_values, gene_names, sample_num, top_num, node)

    # generate bar char
    bar = plot_shap_values(model_name, node, shap_values, x_test, gene_names, "bar", (35, 40),
                           save_bar)  # (width, height)
    # generate scatter chart
    scatter = plot_shap_values(model_name, node, shap_values, x_test, gene_names, "dot", (35, 40), save_scatter)

    return num_features, bar, scatter


def create_shap_tree_val(model_name: str, dm: DataNormalization, phen: ndarray, gene: DataFrame, ids: ndarray,
                         hidden_vars: DataFrame, save_bar: Path, save_scatter: Path, gene_model: Path,
                         num_workers: int, fold: bool, test_split: float, random_state: int, shuffle: bool,
                         boost: bool, top_rate: float):
    create_dir(save_bar)
    create_dir(save_scatter)
    create_dir(gene_model)
    column_num: int = len(hidden_vars.columns)
    sample_num: int = len(gene.index)
    top_num: int = int(top_rate * len(gene.columns))

    result = []
    unique, unique_count = np.unique(phen, return_counts=True)

    # create a Table with the same columns as above,
    # plus confidence scores for all labels
    columns = ["Node", "SHAP Summary Plot - Bar\nTop 20", "SHAP Summary Plot - Scatter\nTop 20 Features",
               "Number of Features", "R2 Score"]
    summary_tbl: Table = wandb.Table(columns=columns)

    params = ((boost, phen, unique, unique_count, gene, hidden_vars[i], test_split, shuffle,
               random_state, num_workers, dm, fold, sample_num, ids, top_num, gene_model,
               model_name, save_bar, save_scatter, column_num, i, summary_tbl) for i in range(column_num))
    for r in map(lambda p: predict_shap_values(*p), params):
        result.append(r)

    r2_scores = pd.DataFrame(result, columns=['node', 'R^2'])
    r2_scores.to_csv(f'{gene_model}-r2.csv', header=True, index=False)
    tmp = f"{model_name}-summary-r2)"
    wandb.log({tmp: summary_tbl})


def main(model_name, gene_name, gene_id, ae_result, col_mask, save_dir: Path, save_bar, save_scatter, gene_model,
         num_workers, fold, test_split, random_state, shuffle, boost: bool, top_rate):
    gene: DataFrame

    with wandb.init(name=model_name, project="XAE4Exp"):
        # wandb configuration
        wandb.config.update = {"architecture": platform.platform(),
                               "Note": f"stratify splitting if there more than 2 phenotype and each category has more "
                                       f"than 1 element; random_state={random_state}"}
        save_bar = save_dir.joinpath(save_bar)
        save_scatter = save_dir.joinpath(save_scatter)
        gene_model = save_dir.joinpath(gene_model)
        create_dir(save_bar.parent)
        create_dir(save_scatter.parent)
        create_dir(gene_model.parent)
        geno_id: DataFrame

        mask: pd.DataFrame = get_data(col_mask)
        geno_id, phen = get_phen(get_data(gene_id))

        if gene_name is None:
            gene = convert_gene_id_to_name(geno_id, mask.columns.to_numpy())
        else:
            gene = get_data(gene_name)

        hidden_vars: DataFrame = get_data(ae_result, index_col=None, header=None)
        ids: ndarray = geno_id.columns.to_numpy()[mask.values.flatten()]
        print(f"\ndf mask:\n{mask}\nnp mask:\n{mask.to_numpy()}\n")
        print(f"gene: {gene.shape}\n{gene.columns.to_numpy()}")
        # dm = DataNormalization(column_mask=mask.to_numpy().flatten(), column_names=gene.columns.to_numpy())
        dm = DataNormalization(column_mask=mask.to_numpy().flatten())
        create_shap_tree_val(model_name, dm, phen.to_numpy(), gene, ids, hidden_vars, save_bar, save_scatter,
                             gene_model, num_workers, fold, test_split, random_state, shuffle, boost, top_rate)

        wandb.finish()


def add_shap_arguments(parse):
    parse.add_argument("-b", "--save_bar", type=str, required=True,
                       help='base dir to saved AE models e.g. ./shap/bar')
    parse.add_argument("--save_scatter", type=str, required=True,
                       help='path to save SHAP scatter chart e.g. ./shap/scatter')
    parse.add_argument("-m", "--gene_model", type=str, required=True,
                       help='path to save gene module e.g. ./shap/gene_model')
    parse.add_argument("-tr", "--top_rate", type=float, default=0.2,
                       help='test set split ratio. default is 0.2')
    parse.add_argument("--boost", action='store_true', default=False,
                       help="whether to use RandomForrestRegressor or XGBRegressor. default is")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="calculates the shapey values for the AE model's output")
    add_shap_arguments(parser)
    parser.add_argument("-sd", "--save_dir", type=Path,
                        default=Path(__file__).absolute().parent.parent.joinpath("AE"),
                        help='base dir to saved Shap models e.g. ./AE/shap')
    parser.add_argument("--ae_result", type=Path, required=True,
                        help='path to AutoEncoder results.  e.g. ./AE_199.csv')
    parser.add_argument("--col_mask", type=Path, required=True,
                        help='path to column mask data.')
    parser.add_argument("--gene_name", type=Path,
                        help='path to input data with gene name as column headers e.g. ./gene_name_QC')
    parser.add_argument("-id", "--gene_id", type=Path, required=True,
                        help='path to input data with gene id as column headers e.g. ./gene_id_QC')
    parser.add_argument("--model_name", type=str, required=True, help='AE model name')
    parser.add_argument("-w", "--num_workers", type=int,
                        help='number of processors used to run in parallel. -1 mean using all processor '
                             'available default is None')
    parser.add_argument("--fold", action="store_true", default=False,
                        help='selecting this flag causes the x to be transformed to change fold relative to '
                             'row median. default is False')
    parser.add_argument("-ts", "--test_split", type=float, default=0.2,
                        help='test set split ratio. default is 0.2')
    parser.add_argument("-rs", "--random_state", type=int, default=42,
                        help='sets a seed to the random generator, so that your train-val-test splits are '
                             'always deterministic. default is 42')
    parser.add_argument("-s", "--shuffle", action='store_true', default=False,
                        help='when this flag is used the dataset is shuffled before splitting the dataset.')

    args: Dict[str, Any] = vars(parser.parse_args())
    print(f"args:\n{args}")

    main(**args)
