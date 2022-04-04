# Use Python to plot SHAP figure (include both bar chart and scatter chart) and generate gene module based on SHAP value
import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from pandas import DataFrame
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from CommonTools import create_dir


def get_last_model(directory: Path):
    file_path: Path
    time: float
    time, file_path = max((f.stat().st_mtime, f) for f in directory.iterdir())
    return file_path


def main(model_name, gene_name, gene_id, ae_result, save_bar, save_scatter, gene_model, fold, test_split, random_state,
         shuffle):
    # TODO need to refactor this method to incorporate the changes in x normalization
    create_dir(Path(save_bar).parent)
    create_dir(Path(save_scatter).parent)
    create_dir(Path(gene_model).parent)
    gene: get_gene_name_data(gene_name, )

    DataFrame = pd.read_csv(gene_name, index_col=0)
    gene = get_normalized_data(gene)
    hidden_vars: DataFrame = pd.read_csv(ae_result, header=None)
    column_num: int = len(hidden_vars.columns)
    sample_num: int = len(gene.index)
    top_rate: float = 1 / 20  # top rate of gene columns
    top_num: int = int(top_rate * len(gene.columns))
    gene_id: DataFrame = pd.read_csv(gene_id, index_col=0, header=None)
    gene_id: np.ndarray = np.array(gene_id.columns)

    for i in range(column_num):
        x_train: Any
        x_test: Any
        y_train: Any
        y_test: Any
        x_train, x_test, y_train, y_test = train_test_split(gene,
                                                            hidden_vars[i],
                                                            test_size=0.2,
                                                            random_state=42)
        my_model: RandomForestRegressor = RandomForestRegressor(bootstrap=True, oob_score=False, max_depth=20,
                                                                random_state=42, n_estimators=100)
        my_model.fit(x_train, y_train)
        explainer = shap.TreeExplainer(my_model)
        # **explainer = shap.KernelExplainer(my_model.predict, x = x_test.iloc[0:10])
        shap_values = explainer.shap_values(x_test)
        # **generate gene model
        shap_values_mean = np.sum(abs(shap_values), axis=0) / sample_num
        shap_values_ln = np.log(shap_values_mean)  # *calculate ln^|shap_values_mean|
        gene_module = np.stack((gene_id, shap_values_ln), axis=0)
        gene_module = gene_module.T
        gene_module = gene_module[np.argsort(gene_module[:, 1])]
        gene_module = gene_module[::-1]
        gene_model: DataFrame = pd.DataFrame(gene_module)
        gene_model = gene_model.head(top_num)
        gene_model = gene_model[(gene_model[[1]] != -np.inf).all(axis=1)]
        if len(gene_model.index) > (1 / 4) * top_num:
            print(f'{gene_model}({i}).csv')
            gene_model.to_csv(f'{gene_model}({i}).csv', header=True, index=False, sep='\t')
        # generate bar chart
        shap.summary_plot(shap_values, x_test, plot_type='bar', plot_size=(15, 10))
        print(f'{save_bar}({i}).png')
        plt.savefig(f'{save_bar}({i}).png', dpi=100, format='png')
        plt.close()
        # generate scatter chart
        shap.summary_plot(shap_values, x_test, plot_size=(15, 10))
        print(f'{save_scatter}({i}).png')
        plt.savefig(f'{save_scatter}({i}).png', dpi=100, format='png')
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="calculates the shapey values for the AE model's output")

    parser.add_argument("--model_name", type=str, help='AE model name')
    parser.add_argument("-name", "--gene_name", type=Path,
                        help='path to input data with gene name as column headers e.g. ./gene_name_QC')
    parser.add_argument("-id", "--gene_id", type=Path, required=True,
                        help='path to input data with gene id as column headers e.g. ./gene_id_QC')
    parser.add_argument("--ae_result", type=Path, required=True,
                        help='path to AutoEncoder results.  e.g. ./AE_199.csv')
    parser.add_argument("--column_mask", type=Path, required=True,
                        help='path to column mask data.')
    parser.add_argument("-b", "--save_bar", type=Path,
                        default=Path(__file__).absolute().parent.parent.joinpath("shap/bar"),
                        help='base dir to saved AE models e.g. ./shap/bar')
    parser.add_argument("--save_scatter", type=Path,
                        default=Path(__file__).absolute().parent.parent.joinpath("shap/scatter"),
                        help='path to save SHAP scatter chart e.g. ./shap/scatter')
    parser.add_argument("-m", "--gene_model", type=Path,
                        default=Path(__file__).absolute().parent.parent.joinpath("shap/bar"),
                        help='path to save gene module e.g. ./shap/gene_model')
    parser.add_argument("--fold", type=bool, default=False,
                        help='selecting this flag causes the x to be transformed to change fold relative to '
                             'row median. default is False')
    parser.add_argument("-ts", "--test_split", type=float, default=0.2,
                        help='test set split ratio. default is 0.2')
    parser.add_argument("-w", "--num_workers", type=int, default=0,
                        help='number of processors used to load x. ie worker = 4 * # of GPU. default is 0')
    parser.add_argument("-rs", "--random_state", type=int, default=42,
                        help='sets a seed to the random generator, so that your train-val-test splits are '
                             'always deterministic. default is 42')
    parser.add_argument("-s", "--shuffle", action='store_true', default=False,
                        help='when this flag is used the dataset is shuffled before splitting the dataset.')

    args: argparse.Namespace = parser.parse_args()
    main(args)
