import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
import shap

RNA_name = 'BRCA'
compress_num = '12'
gene = pd.read_csv()
hidden_vars = pd.read_csv()
column_num = len(hidden_vars.columns)

for i in range(column_num):
  X_train, X_test, Y_train, Y_test = train_test_split(gene,
                                                hidden_vars[i],
                                                test_size=0.2,
                                                random_state=42)
  my_model = RandomForestRegressor(bootstrap=True, oob_score=False,max_depth=20, random_state=42, n_estimators=100)
  my_model.fit(X_train, Y_train)
  explainer = shap.TreeExplainer(my_model)
  #explainer = shap.KernelExplainer(my_model.predict,data = X_test)
  shap_values = explainer.shap_values(X_test)
  shap.summary_plot(shap_values, X_test, plot_size = (10,10))
  plt.savefig(, format='pdf')
  plt.close()
