import numpy as np
import pandas as pd

threshold = 0.05
gene = pd.read_csv('/export/home/yang.yu2/UK_Biobank_code/figure/BRCA_20000.txt',sep = '\t',header = None)
geneid = pd.DataFrame(columns = [0,1])

for i in range(len(gene)):
  if gene.iloc[i,1]< threshold:
    geneid = geneid.append(gene.iloc[i],ignore_index = True)

geneid = geneid[0]
geneid.to_csv('/export/home/yang.yu2/UK_Biobank_code/figure/ORA/BRCA_20000_'+str(threshold)+'.txt',header = 0, index = 0, sep = '\t')
