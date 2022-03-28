## Use Python for x preprocessing (quality control)
## path - is a string to desired path location. 

import numpy as np
import pandas as pd

PATH_TO_DATA = './x.txt'           #path to x (before quality control)
PATH_TO_SAVE = './data_QC.txt'       #path to save x after quality control
 
Tumor1 = pd.read_csv(PATH_TO_DATA,index_col=0)
Tumor1_var = Tumor1.var()
Tumor1.drop(Tumor1_var[Tumor1_var < 1].index.values, axis=1, inplace=True)
