# path - is a string to desired path location.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
import shap

PATH_TO_DATA = './data_QC.txt'    #path to cleaned data (after quality control)
PATH_TO_AE_RESULT = './AE_199.txt'    #path to AutoEncoder results, alwarys the last epoch result
PATH_TO_SAVE = './shap.txt'.          #path to save shap results

RNA_name = 'BRCA' #data file name
compress_num = '4000' #AutoEncoder compress rate
total_value = 0
gene = pd.read_csv(PATH_TO_DATA,index_col=0)
hidden_vars = pd.read_csv(PATH_TO_AE_RESULT,header = None)
column_num = len(hidden_vars.columns)

for i in range(column_num):
  X_train, X_test, Y_train, Y_test = train_test_split(gene,
                                                hidden_vars[i],
                                                test_size=0.2,
                                                random_state=42)
  my_model = RandomForestRegressor(bootstrap=True, oob_score=False,max_depth=20, random_state=42, n_estimators=100)
  my_model.fit(X_train, Y_train)
  shap_values = shap.TreeExplainer(my_model).shap_values(X_test)
  shap_values_sum = np.sum(shap_values,axis=0)
  print(shap_values_sum)
  total_value = total_value + shap_values_sum

gene_id = pd.read_csv(PATH_TO_DATA,index_col=0,header=None)
gene_id = gene_id.iloc[0]
gene_id = gene_id.to_numpy()

enrich_analysis = np.stack((gene_id,total_value),axis=0)
enrich_analysis = enrich_analysis.T
enrich_analysis = enrich_analysis[np.argsort(enrich_analysis[:,1])]
enrich_analysis = np.flip(enrich_analysis,0)
enrich_analysis = pd.DataFrame(enrich_analysis)

enrich_analysis.to_csv(PATH_TO_SAVE,header = 0, index = 0, sep = '\t')
