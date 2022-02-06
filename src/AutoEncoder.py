# ** Use Python to run Deep Autoencoder (feature selection) **
# ** path - is a string to desired path location. **
import math
import sys
from typing import Union, Any

import pandas as pd
import sklearn.preprocessing
import torch
from pathlib import Path
from pandas import Series, DataFrame
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from AutoEncoderModule import GPDataSet
from AutoEncoderModule import AutoGenoShallow
from AutoEncoderModule import run_ae


def get_filtered_data(geno, path_to_save_qc: Path) -> DataFrame:
    geno_var: Any = geno.var()
    geno.drop(geno_var[geno_var < 1].index.values, axis=1, inplace=True)
    sklearn.preprocessing.minmax_scale(X=geno, feature_range=(0, 1), axis=0, copy=False)
    geno.to_csv(path_to_save_qc)
    return geno


def main(model_name: str, path_to_data: Path, path_to_save_qc: Path, path_to_save_ae: Path, compression_ratio: int):
    if not (path_to_save_ae.is_dir()):
        print(f'{path_to_save_ae} is not a directory')
        sys.exit(-1)

    if not (path_to_data.is_file()):
        print(f'{path_to_data} is not a file')
        sys.exit(-1)

    # data quality control
    geno = get_filtered_data(pd.read_csv(path_to_data, index_col=0), path_to_save_qc)

    batch_size = 4096
    geno_train, geno_test = train_test_split(geno, test_size=0.1, random_state=42)

    geno_train_set: GPDataSet = GPDataSet(geno)
    geno_train_set_loader = DataLoader(dataset=geno_train_set, batch_size=batch_size, shuffle=False, num_workers=8)

    geno_test_set = GPDataSet(geno_test)
    geno_test_set_loader = DataLoader(dataset=geno_test_set, batch_size=batch_size, shuffle=False, num_workers=8)
    input_features = len(geno.columns)
    output_features = input_features
    smallest_layer = math.ceil(input_features / compression_ratio)
    hidden_layer = int(2 * smallest_layer)

    model = AutoGenoShallow(input_features=input_features, hidden_layer=hidden_layer,
                            smallest_layer=smallest_layer, output_features=output_features).cuda()
    distance = nn.MSELoss()  # for regression, 0, 0.5, 1
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    run_ae(model_name=model_name, model=model, geno_train_set_loader=geno_train_set_loader,
           geno_test_set_loader=geno_test_set_loader, input_features=input_features,
           optimizer=optimizer, distance=distance, do_train=True, do_test=True, save_dir=path_to_save_ae)


if __name__ == '__main__':
    if len(sys.argv) < 5:
        print('less than 6 command line arguments')
        print('python AutoEncoder.py model_name original_datafile '
              'quality_control_filename dir_AE_model compression_ratio')
        print('\tmodel_name - model name e.g. AE_Geno')
        print('\toriginal_datafile - original datafile e.g. ./data/data_example.csv')
        print('\tquality_control_filename - filename of original data after quality control e.g. ./data/data_QC.csv')
        print('\tdir_AE_model - base dir to saved AE models e.g. .data/filter/AE')
        print('\tcompression_ratio - compression ratio for smallest layer NB: ideally a number that is power of 2')
        sys.exit(-1)

    main(sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), int(sys.argv[5]))
