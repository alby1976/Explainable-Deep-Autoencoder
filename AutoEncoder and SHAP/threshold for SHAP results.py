# path - is a string to desired path location. The file does

import numpy as np
import pandas as pd

PATH_TO_SHAP_RESULT = './shap.txt'    #path to SHAP results
PATH_TO_SAVE = './shap_new.txt'     #path to save SHAP results after threshold

threshold = 0.05
gene = pd.read_csv(PATH_TO_SHAP_RESULT,sep = '\t',header = None)
geneid = pd.DataFrame(columns = [0,1])

for i in range(len(gene)):
  if gene.iloc[i,1]< threshold:
    geneid = geneid.append(gene.iloc[i],ignore_index = True)

geneid = geneid[0]
geneid.to_csv(PATH_TO_SAVE,header = 0, index = 0, sep = '\t')
