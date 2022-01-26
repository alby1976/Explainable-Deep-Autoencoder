## Use Python to run Deep Autoencoder (feature selection)
## path - is a string to desired path location.

import pandas as pd
import numpy as np
import torch
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pickle
import os
import time

PATH_TO_DATA = 'data_example.csv'      #path to original data 
PATH_TO_SAVE_AE = '/example/'      #path to save AutoEncoder results
PATH_TO_SAVE_QC = 'data_example_QC.csv'       #path to save original data after quality control

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

batch_size = 100
geno_train, geno_test = train_test_split(geno, test_size=0.1, random_state=42)


class GPDataSet(Dataset):
    def __init__(self, GP_list):
        # 'Initialization'
        self.GP_list = GP_list

    def __len__(self):
        # 'Denotes the total number of samples'
        return len(self.GP_list)

    def __getitem__(self, index):
        # 'Generates one sample of data'
        # Load data and get label
        X = self.GP_list[index]
        X = np.array(X)
        return X


geno_train_set = GPDataSet(geno)
geno_train_set_loader = DataLoader(dataset=geno_train_set, batch_size=batch_size, shuffle=False, num_workers=8)

geno_test_set = GPDataSet(geno_test)
geno_test_set_loader = DataLoader(dataset=geno_test_set, batch_size=batch_size, shuffle=False, num_workers=8)
input_features = int(snp)
output_features = input_features
smallest_layer = int(snp / 200)
hidden_layer = int(2*smallest_layer)

class Auto_Geno_shallow(nn.Module):
    def __init__(self):
        super(Auto_Geno_shallow, self).__init__()  # I guess this inherit __init__ from super class

        # def the encoder function
        self.encoder = nn.Sequential(
            nn.Linear(input_features, hidden_layer),
            nn.ReLU(True),
            nn.Linear(hidden_layer, smallest_layer),
            nn.ReLU(True),
        )

        # def the decoder function
        self.decoder = nn.Sequential(
            nn.Linear(smallest_layer, hidden_layer),
            nn.ReLU(True),
            nn.Linear(hidden_layer, output_features),
            nn.Sigmoid()
        )

    # def forward function
    def forward(self, x):
        y = self.encoder(x)
        x = self.decoder(y)
        return x, y


model = Auto_Geno_shallow().cuda()
distance = nn.MSELoss()  # for regression, 0, 0.5, 1
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
do_train = True
do_test = True
num_epochs = 200
for epoch in range(num_epochs):
    start_time = time.time()
    batch_precision_list = []
    output_coder_list = []
    if do_train:
        sum_loss = 0
        current_batch = 0
        model.train()
        for geno_data in geno_train_set_loader:
            current_batch += 1
            train_geno = Variable(geno_data).float().cuda()
            # =======forward========
            output, coder = model.forward(train_geno)
            loss = distance(output, train_geno)
            sum_loss += loss.item()
            # ======get coder======
            coder2 = coder.cpu().detach().numpy()
            output_coder_list.extend(coder2)
            # ======precision======
            output2 = output.cpu().detach().numpy()
            output3 = np.floor(output2 * 3) / 2  # make output3's value to 0, 0.5, 1
            diff = geno_data.numpy() - output3  # [0,0.5,1] - [0.0, 0.5, 0.5]
            diff_num = np.count_nonzero(diff)
            batch_average_precision = 1 - diff_num / (batch_size * input_features)
            batch_precision_list.append(batch_average_precision)
            # ======backward========
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===========log============
        coder_np = np.array(output_coder_list)
        temp = round(smallest_layer / 100)
        coder_file = save_dir + model_name + str(epoch) + '.csv'
        np.savetxt(fname=coder_file, X=coder_np, fmt='%f', delimiter=',')
        print('epoch[{}/{}],loss:{:.4f}'.format(epoch + 1, num_epochs, sum_loss))
        average_precision = sum(
            batch_precision_list) / current_batch  # precision_list is a list of [ave_pre_batch1, ave_pre_batch2,...]
        print('precision: ' + str(average_precision))
    # ===========test==========
    test_batch_precision_list = []
    if do_test:
        test_sum_loss = 0
        test_current_batch = 0
        model.eval()
        for geno_test_data in geno_test_set_loader:
            test_current_batch += 1
            test_geno = Variable(geno_test_data).float().cuda()
            # =======forward========
            test_output, coder = model.forward(test_geno)
            loss = distance(test_output, test_geno)
            test_sum_loss += loss.item()
            # ======precision======
            test_output2 = test_output.cpu().detach().numpy()
            test_output3 = np.floor(test_output2 * 3) / 2  # make output3's value to 0, 0.5, 1
            diff = geno_test_data.numpy() - test_output3  # [0,0.5,1] - [0.0, 0.5, 0.5]
            diff_num = np.count_nonzero(diff)
            batch_average_precision = 1 - diff_num / (batch_size * input_features)  # a single value
            test_batch_precision_list.append(batch_average_precision)  # [ave_pre_batch1, ave_pre_batch2,...]
    test_average_precision = sum(
        test_batch_precision_list) / test_current_batch  # precision_list is a list of [ave_pre_batch1, ave_pre_batch2,...]
    print('test precision: ' + str(test_average_precision))
