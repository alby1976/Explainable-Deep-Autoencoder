# Use Python to plot SHAP figure (include both bar chart and scatter chart) and generate gene module based on SHAP value
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from pandas import DataFrame
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from AutoEncoderModule import create_dir


def get_last_model(directory: Path):
    file_path: Path
    time: float
    time, file_path = max((f.stat().st_mtime, f) for f in directory.iterdir())
    return file_path


def main(path_to_data_gene_name: Path, path_to_data_gene_id: Path, path_to_ae_result: Path,
         path_to_save_bar: str, path_to_save_scatter: str, path_to_save_gene_model: str):
    create_dir(Path(path_to_save_bar).parent)
    create_dir(Path(path_to_save_scatter).parent)
    create_dir(Path(path_to_save_gene_model).parent)
    gene: DataFrame = pd.read_csv(path_to_data_gene_name, index_col=0)
    hidden_vars: DataFrame = pd.read_csv(path_to_ae_result, header=None)
    column_num: int = len(hidden_vars.columns)
    sample_num: int = len(gene.index)
    gene = minmax_scale(X=gene, feature_range=(0, 1), axis=0, copy=True)
    top_rate: float = 1 / 20  # top rate of gene columns
    top_num: int = int(top_rate * len(gene.columns))
    gene: np.ndarray = minmax_scale(X=gene, feature_range=(0, 1), axis=0, copy=True)
    gene_id: DataFrame = pd.read_csv(path_to_data_gene_id, index_col=0, header=None)
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
        # **explainer = shap.KernelExplainer(my_model.predict, data = x_test.iloc[0:10])
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
            print(f'{path_to_save_gene_model}({i}).csv')
            gene_model.to_csv(f'{path_to_save_gene_model}({i}).csv', header=True, index=False, sep='\t')
        # generate bar chart
        shap.summary_plot(shap_values, x_test, plot_type='bar', plot_size=(15, 10))
        print(f'{path_to_save_bar}({i}).png')
        plt.savefig(f'{path_to_save_bar}({i}).png', dpi=100, format='png')
        plt.close()
        # generate scatter chart
        shap.summary_plot(shap_values, x_test, plot_size=(15, 10))
        print(f'{path_to_save_scatter}({i}).png')
        plt.savefig(f'{path_to_save_scatter}({i}).png', dpi=100, format='png')
        plt.close()


if __name__ == '__main__':
    if len(sys.argv) < 6:
        print('Default setting are used. Either change SHAP_combo.py to change settings or type:\n')
        print('python SHAP_combo.py path_to_data_gene_name PATH_TO_DATA_GENE_ID PATH_TO_AE_RESULT '
              'PATH_TO_SAVE_BAR PATH_TO_SAVE_SCATTER PATH_TO_SAVE_GENE_MODULE')
        print('\tpath_to_data_gene_name - path to cleaned data with gene name after quality control '
              'e.g. ./gene_name_QC.csv')
        print('\tPATH_TO_DATA_GENE_ID - path to cleaned data with gene id after quality control e.g ./gene_id_QC.csv')
        print('\tPATH_TO_AE_RESULT - path to AutoEncoder results, always the last epoch result e.g. ./AE_199.csv')
        print('\tPATH_TO_SAVE_BAR - path to save SHAP bar chart e.g. ./shap/bar')
        print('\tPATH_TO_SAVE_SCATTER - path to save SHAP scatter chart e.g. ./shap/scatter')
        print('\tPATH_TO_SAVE_GENE_MODEL - path to save gene module e.g. ./shap/gene_model')
        main(Path('./gene_name_QC.csv'), Path('./gene_id_QC.csv'), Path('./AE_199.csv'),
             './shap/bar', './shap/scatter', './shap/gene_module')
    else:

        main(Path(sys.argv[1]), Path(sys.argv[2]), get_last_model(Path(sys.argv[3])),
             sys.argv[4], sys.argv[5], sys.argv[6])
