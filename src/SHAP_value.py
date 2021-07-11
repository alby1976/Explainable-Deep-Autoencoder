## Use Python to generate gene module based on SHAP value

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
import shap

PATH_TO_DATA = './data_QC.txt'    #path to cleaned data (after quality control)
PATH_TO_AE_RESULT = './AE_199.txt'    #path to AutoEncoder results, alwarys the last epoch result
PATH_TO_SAVE = './shap.txt'.          #path to save gene module

gene_id = pd.read_csv(PATH_TO_DATA, index_col=0)
hidden_vars = pd.read_csv(PATH_TO_AE_RESULT, header = None)
column_num = len(hidden_vars.columns)       # column number of AE results
sample_num = len(gene_id.index)             # sample number of data
top_num = int((1/20)*len(gene_id.columns))  # threshold for row number of gene module

for i in range(column_num):
  X_train, X_test, Y_train, Y_test = train_test_split(gene_id,
                                                hidden_vars[i],
                                                test_size=0.2,
                                                random_state=42)
  my_model = RandomForestRegressor(bootstrap=True, oob_score=False,max_depth=20, random_state=42, n_estimators=100)
  my_model.fit(X_train, Y_train)
  explainer = shap.TreeExplainer(my_model)
  #explainer = shap.KernelExplainer(my_model.predict, data = X_test.iloc[0:10])
  shap_values = explainer.shap_values(X_test)
  ## generate gene module
  shap_values_mean = np.sum(abs(shap_values),axis=0)/sample_num
  shap_values_ln = np.log(shap_values_mean) ##calculate ln^|shap_values_mean|
  gene_module = np.stack((gene_id,shap_values_ln),axis=0)
  gene_module = gene_module.T
  gene_module = gene_module[np.argsort(gene_module[:,1])]
  gene_module = gene_module[::-1]
  gene_module = pd.DataFrame(gene_module)
  gene_module = gene_module.head(top_num)
  gene_module = gene_module[(gene_module[[1]]!= -np.inf).all(axis=1)]
  if len(gene_module.index) > (0.10)*top_num:
      gene_module.to_csv(PATH_TO_SAVE, header = 0, index = 0, sep = '\t')
