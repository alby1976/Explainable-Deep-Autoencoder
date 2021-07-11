## Use Python to plot SHAP scatter chart

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
import shap

PATH_TO_DATA_GENE_NAME = './gene_name_QC.txt'    # path to cleaned data with gene annotation (not gene id) (after quatlity control)
PATH_TO_AE_RESULT = './AE_199.txt'    # path to AutoEncoder results, alwarys the last epoch result
PATH_TO_SAVE_FIGURE = '.figure.pdf'     # path to save SHAP figure


gene = pd.read_csv(PATH_TO_DATA_GENE_NAME)
hidden_vars = pd.read_csv(PATH_TO_AE_RESULT)
column_num = len(hidden_vars.columns)

for i in range(column_num):
  X_train, X_test, Y_train, Y_test = train_test_split(gene,
                                                hidden_vars[i],
                                                test_size=0.2,
                                                random_state=42)
  my_model = RandomForestRegressor(bootstrap=True, oob_score=False,max_depth=20, random_state=42, n_estimators=100)
  my_model.fit(X_train, Y_train)
  explainer = shap.TreeExplainer(my_model)
  #explainer = shap.KernelExplainer(my_model.predict,data = X_test) # different SHAP explainer could be used
  shap_values = explainer.shap_values(X_test)
  shap.summary_plot(shap_values, X_test, plot_size = (10,10))
  plt.savefig(PATH_TO_SAVE_FIGURE, format='pdf')
  plt.close()
