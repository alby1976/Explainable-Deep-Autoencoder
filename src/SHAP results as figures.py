# path - is a string to desired path location.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
import shap

PATH_TO_DATA = './data_QC.txt'    #path to cleaned data (after quatlity control)
PATH_TO_AE_RESULT = './AE_199.txt'    #path to AutoEncoder results, alwarys the last epoch result
PATH_TO_SAVE_FIGURE = '.figure.pdf'     #path to save SHAP figure
RNA_name = 'BRCA'
gene = pd.read_csv(PATH_TO_DATA ,index_col=0)
hidden_vars = pd.read_csv(PATH_TO_AE_RESULT,header = None)

X_train, X_test, Y_train, Y_test = train_test_split(gene,
                                                hidden_vars[0],
                                                test_size=0.1,
                                                random_state=42)

my_model = RandomForestRegressor(bootstrap=True, oob_score=False, max_depth=20, random_state=42, n_estimators=100)

my_model.fit(X_train, Y_train)
shap_values = shap.TreeExplainer(my_model).shap_values(X_test)

shap.summary_plot(shap_values, X_test, plot_type="bar")
plt.savefig(PATH_TO_SAVE_FIGURE, format='pdf')
