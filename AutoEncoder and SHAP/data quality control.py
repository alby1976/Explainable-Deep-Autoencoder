import numpy as np
import pandas as pd
Tumor = pd.read_csv('/export/home/yang.yu2/UK_Biobank_data/TumorOnly/BRCA_TumorOnly.csv',index_col=0)
Tumor_var = Tumor.var()
for i in range(len(Tumor_var)-1,-1,-1):
  if Tumor_var[i]<1:
    del Tumor[Tumor.columns[i]]
Tumor.to_csv('/export/home/yang.yu2/UK_Biobank_data/TumorOnly_QC/BRCA_TumorOnly_QC.csv')