## Use Python to plot SHAP figure (include both bar chart and scatter chart) and generate gene module based on SHAP value
from typing import Union, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from pandas import Series, DataFrame
from pandas.io.parsers import TextFileReader
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Input
PATH_TO_DATA_GENE_NAME = './gene_name_QC.csv'  # path to cleaned data with gene annotation (not gene id) (after quatlity control)
PATH_TO_DATA_GENE_ID = './gene_id_QC.csv'  # path to cleaned data with gene id (not gene name) (after quality control)
PATH_TO_AE_RESULT = './AE_199.csv'  # path to AutoEncoder results, always the last epoch result

# Output
PATH_TO_SAVE_BAR: str = './shap/bar'  # path to save SHAP bar chart
PATH_TO_SAVE_SCATTER: str = './shap/scatter'  # path to save SHAP scatter chart
PATH_TO_SAVE_GENE_MODULE: str = './shap/gene_module'  # path to save gene module

gene: Union[Union[TextFileReader, Series, DataFrame, None], Any] = pd.read_csv(PATH_TO_DATA_GENE_NAME, index_col=0)
hidden_vars: Union[Union[TextFileReader, Series, DataFrame, None], Any] = pd.read_csv(PATH_TO_AE_RESULT, header=None)
column_num: int = len(hidden_vars.columns)
sample_num: int = len(gene.index)
top_rate: float = 1 / 20  # top rate of gene columns
top_num: int = int(top_rate * len(gene.columns))
gene_id: DataFrame = pd.read_csv(PATH_TO_DATA_GENE_ID, index_col=0, header=True)
gene_id = np.array(gene_id.columns)

for i in range(column_num):
    X_train, X_test, Y_train, Y_test = train_test_split(gene,
                                                        hidden_vars[i],
                                                        test_size=0.2,
                                                        random_state=42)
    my_model = RandomForestRegressor(bootstrap=True, oob_score=False, max_depth=20, random_state=42, n_estimators=100)
    my_model.fit(X_train, Y_train)
    explainer = shap.TreeExplainer(my_model)
    # **explainer = shap.KernelExplainer(my_model.predict, data = X_test.iloc[0:10])
    shap_values = explainer.shap_values(X_test)
    # **generate gene module
    shap_values_mean = np.sum(abs(shap_values), axis=0) / sample_num
    shap_values_ln = np.log(shap_values_mean)  # *calculate ln^|shap_values_mean|
    gene_module = np.stack((gene_id, shap_values_ln), axis=0)
    gene_module = gene_module.T
    gene_module = gene_module[np.argsort(gene_module[:, 1])]
    gene_module = gene_module[::-1]
    gene_module = pd.DataFrame(gene_module)
    gene_module = gene_module.head(top_num)
    gene_module = gene_module[(gene_module[[1]] != -np.inf).all(axis=1)]
    if len(gene_module.index) > (1 / 4) * top_num:
        gene_module.to_csv(PATH_TO_SAVE_GENE_MODULE + str(i) + '.csv', header=0, index=0, sep='\t')
    ## generate bar chart
    shap.summary_plot(shap_values, X_test, plot_type='bar', plot_size=(15, 10))
    plt.savefig(PATH_TO_SAVE_BAR + str(i) + '.png', dpi=100, format='png')
    plt.close()
    ## generate scatter chart
    shap.summary_plot(shap_values, X_test, plot_size=(15, 10))
    plt.savefig(PATH_TO_SAVE_SCATTER + str(i) + '.png', dpi=100, format='png')
    plt.close()
