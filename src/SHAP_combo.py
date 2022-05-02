# Use Python to plot SHAP figure (include both bar chart and scatter chart) and generate gene module based on SHAP value
import argparse
import platform
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import shap
import sklearn
import wandb
from numpy import ndarray
from pandas import DataFrame
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from CommonTools import create_dir, get_phen, get_data, DataNormalization, \
    convert_gene_id_to_name
from src.ShapDeepExplainerModule import process_shap_values


def get_last_model(directory: Path):
    file_path: Path
    time: float
    time, file_path = max((f.stat().st_mtime, f) for f in directory.iterdir())
    return file_path


def predict_shap_values(phen, unique, unique_count, gene, hidden_vars, test_split, shuffle, random_state,
                        num_workers, dm, fold, sample_num, ids, top_num, gene_model, model_name, save_bar,
                        save_scatter, column_num, i) -> Tuple[int, float]:
    print(f'**** Processing {i + 1} out of {column_num} columns ****')
    x_train: Any
    x_test: Any
    y_train: Any
    y_test: Any
    phen_train: Any = None
    phen_test: Any
    if phen is not None and unique.size > 1 and np.min(unique_count) > 1:
        x_train, x_test, y_train, y_test, phen_train, phen_test = train_test_split(gene.to_numpy(),
                                                                                   hidden_vars,
                                                                                   phen.to_numpy(),
                                                                                   test_size=test_split,
                                                                                   stratify=phen,
                                                                                   random_state=random_state)
    else:
        x_train, x_test, y_train, y_test, = train_test_split(gene,
                                                             hidden_vars,
                                                             test_size=test_split,
                                                             shuffle=shuffle,
                                                             stratify=phen,
                                                             random_state=random_state)

    my_model: RandomForestRegressor = RandomForestRegressor(bootstrap=True, oob_score=False, max_depth=20,
                                                            random_state=random_state, n_estimators=100,
                                                            n_jobs=num_workers)
    print(f"\nx_train: {x_train.shape} y_train: {y_train.shape} phen_train: {phen_train.shape}")
    dm.fit(x_train, fold)
    my_model.fit(dm.transform(x_train, fold), y_train)
    y_pred = my_model.predict(x_test)
    r2: float = sklearn.metrics.r2_score(y_true=y_test, y_pred=y_pred)
    explainer = shap.TreeExplainer(my_model)
    # **explainer = shap.KernelExplainer(my_model.predict, x = x_test.iloc[0:10])
    x_test = dm.transform(x_test, fold)
    shap_values = explainer.shap_values(x_test)
    # process shap values and generate gene model
    process_shap_values(save_bar, save_scatter, gene_model, model_name, x_test, shap_values, ids, sample_num,
                        top_num, i)

    return i, r2


def main(model_name, gene_name, gene_id, ae_result, col_mask, save_bar, save_scatter, gene_model, num_workers, fold,
         test_split, random_state, shuffle, top_rate):
    with wandb.init(name=model_name, project="XAE4Exp"):
        # wandb configuration
        wandb.config.update = {"architecture": platform.platform(),
                               "Note": f"stratify splitting if there more than 2 phenotype and each category has more "
                                       f"than 1 element; random_state={random_state}"}
        create_dir(Path(save_bar).parent)
        create_dir(Path(save_scatter).parent)
        create_dir(Path(gene_model).parent)
        geno_id: DataFrame

        mask: pd.DataFrame = get_data(col_mask)
        geno_id, phen = get_phen(get_data(gene_id))

        if gene_name is None:
            gene = convert_gene_id_to_name(geno_id, mask.columns.to_numpy())
        else:
            gene, phen = get_phen(get_data(gene_name))

        hidden_vars: DataFrame = get_data(ae_result, index_col=None, header=None)
        column_num: int = len(hidden_vars.columns)
        sample_num: int = len(gene.index)
        top_num: int = int(top_rate * len(gene.columns))
        ids: ndarray = geno_id.columns.to_numpy()[mask.values.flatten()]
        unique, unique_count = np.unique(phen, return_counts=True)
        print(f"\ndf mask:\n{mask}\nnp mask:\n{mask.to_numpy()}\n")
        dm = DataNormalization(column_mask=mask.values.flatten(), column_names=gene.columns.to_numpy())

        result = []
        params = ((phen, unique, unique_count, gene, hidden_vars[i], test_split, shuffle,
                   random_state, num_workers, dm, fold, sample_num, ids, top_num, gene_model,
                   model_name, save_bar, save_scatter, column_num, i) for i in range(column_num))
        for r in map(lambda p: predict_shap_values(*p), params):
            result.append(r)

    r2_scores = pd.Dataframe(result, columns=['node', 'R^2'])
    r2_scores.to_csv(f'{gene_model}-r2.csv', header=True, index=False)
    tmp = f"{model_name}-r2)"
    tbl = wandb.Table(dataframe=r2_scores)
    wandb.log({tmp: tbl})
    wandb.finish()


def add_shap_arguments(parse):
    parse.add_argument("--model_name", type=str, required=True, help='AE model name')
    parse.add_argument("-name", "--gene_name", type=Path,
                       help='path to input data with gene name as column headers e.g. ./gene_name_QC')
    parse.add_argument("-id", "--gene_id", type=Path, required=True,
                       help='path to input data with gene id as column headers e.g. ./gene_id_QC')
    parse.add_argument("--ae_result", type=Path, required=True,
                       help='path to AutoEncoder results.  e.g. ./AE_199.csv')
    parse.add_argument("--col_mask", type=Path, required=True,
                       help='path to column mask data.')
    parse.add_argument("-b", "--save_bar", type=Path,
                       default=Path(__file__).absolute().parent.parent.joinpath("shap/bar"),
                       help='base dir to saved AE models e.g. ./shap/bar')
    parse.add_argument("--save_scatter", type=Path,
                       default=Path(__file__).absolute().parent.parent.joinpath("shap/scatter"),
                       help='path to save SHAP scatter chart e.g. ./shap/scatter')
    parse.add_argument("-m", "--gene_model", type=Path,
                       default=Path(__file__).absolute().parent.parent.joinpath("shap/gene_model"),
                       help='path to save gene module e.g. ./shap/gene_model')
    parse.add_argument("-tr", "--top_rate", type=float, default=0.2,
                       help='test set split ratio. default is 0.2')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="calculates the shapey values for the AE model's output")
    add_shap_arguments(parser)
    parser.add_argument("-w", "--num_workers", type=int,
                        help='number of processors used to run in parallel. -1 mean using all processor '
                             'available default is None')
    parser.add_argument("--fold", type=bool, default=False,
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
