## Use Python to run Deep Autoencoder (feature selection)
## path - is a string to desired path location.

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import os
from .AutoEncoderModule import GPDataSet
from .AutoEncoderModule import AutoGenoShallow
from .AutoEncoderModule import run_ae

PATH_TO_DATA = './data.txt'      #path to original data
PATH_TO_SAVE_AE = './AE/'      #path to save AutoEncoder results
PATH_TO_SAVE_QC = './data_QC.txt'       #path to save original data after quality control

model_name = 'AE_Geno'
save_dir = PATH_TO_SAVE_AE
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

geno = pd.read_csv(PATH_TO_DATA,index_col=0) #data quality control
geno_var = geno.var()
geno.drop(geno_var[geno_var < 1].index.values, axis=1, inplace=True)
geno.to_csv(PATH_TO_SAVE_QC)
geno = np.array(geno)
snp = int(len(geno[0]))

batch_size = 4096
geno_train, geno_test = train_test_split(geno, test_size=0.1, random_state=42)


geno_train_set = GPDataSet(geno)
geno_train_set_loader = DataLoader(dataset=geno_train_set, batch_size=batch_size, shuffle=False, num_workers=8)

geno_test_set = GPDataSet(geno_test)
geno_test_set_loader = DataLoader(dataset=geno_test_set, batch_size=batch_size, shuffle=False, num_workers=8)
input_features = int(snp)
output_features = input_features
smallest_layer = int(snp / 20000)
hidden_layer = int(2*smallest_layer)

model = AutoGenoShallow(input_features=input_features, hidden_layer=hidden_layer,
                        smallest_layer=smallest_layer, output_features=output_features).cuda()
distance = nn.MSELoss()  # for regression, 0, 0.5, 1
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
run_ae(model=model, model_name=model_name, input_features= input_features, smallest_layer=smallest_layer,
       geno_train_set_loader=geno_train_set_loader, geno_test_set_loader=geno_test_set_loader, distance=distance,
       optimizer=optimizer, do_train=True, do_test=True)
