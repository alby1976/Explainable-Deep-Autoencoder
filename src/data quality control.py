## Use Python for data preprocessing (quality control)
## path - is a string to desired path location. 

import numpy as np
import pandas as pd

PATH_TO_DATA = './data.txt'           #path to data (before quality control)
PATH_TO_SAVE = './data_QC.txt'       #path to save data after quality control

Tumor = pd.read_csv(PATH_TO_DATA,index_col=0)
Tumor_var = Tumor.var()
for i in range(len(Tumor_var)-1,-1,-1):
  if Tumor_var[i]<1:
    del Tumor[Tumor.columns[i]]
Tumor.to_csv(PATH_TO_SAVE)
