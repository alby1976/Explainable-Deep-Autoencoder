import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
import shap

RNA_name = 'BRCA'
gene = pd.read_csv('/export/home/yang.yu2/UK_Biobank_data/TumorOnly_QC/'+RNA_name+'_TumorOnly_QC.csv',index_col=0)
hidden_vars = pd.read_csv('/export/home/yang.yu2/UK_Biobank_code/pytorch_projects/Linear_AE_Geno/'+RNA_name+'199.csv',header = None)

X_train, X_test, Y_train, Y_test = train_test_split(gene,
                                                hidden_vars[0],
                                                test_size=0.1,
                                                random_state=42)

my_model = RandomForestRegressor(bootstrap=True, oob_score=False, max_depth=20, random_state=42, n_estimators=100)

my_model.fit(X_train, Y_train)
shap_values = shap.TreeExplainer(my_model).shap_values(X_test)

shap.summary_plot(shap_values, X_test, plot_type="bar")
plt.savefig('/export/home/yang.yu2/UK_Biobank_code/figure/shap_BRCA.pdf', format='pdf')