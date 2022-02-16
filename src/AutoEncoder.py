# ** Use Python to run Deep Autoencoder (feature selection) **
# ** path - is a string to desired path location. **
import math
import sys
from typing import Union, Any, Tuple
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from scipy.stats import spearmanr, kstest
from numpy import ndarray
from pandas import Series, DataFrame
from sklearn.preprocessing import minmax_scale
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from AutoEncoderModule import GPDataSet
from AutoEncoderModule import AutoGenoShallow
from AutoEncoderModule import create_dir
from AutoEncoderModule import get_normalized_data


def r2_value(y_true: ndarray, y_pred: ndarray) -> float:
    y_ave = y_true.mean()
    sse: int = (np.square(y_pred - y_ave)).sum()
    ssr: int = (np.square(y_true - y_pred)).sum()
    sst: int = (np.square(y_true - y_ave)).sum()
    if sse / sst == 1 - ssr / sst:
        return sse / sst
    else:
        return 1 - ssr / sst


def get_filtered_data(geno: DataFrame, path_to_save_qc: Path) -> DataFrame:
    geno_var: Union[Series, int] = geno.var()
    geno.drop(geno_var[geno_var < 1].index.values, axis=1, inplace=True)
    create_dir(path_to_save_qc.parent)
    geno.to_csv(path_to_save_qc)
    return geno


def main(model_name: str, path_to_data: Path, path_to_save_qc: Path, path_to_save_ae: Path,
         compression_ratio: int, size: int, batch_size: int):
    if not (path_to_save_ae.is_dir()):
        print(f'{path_to_save_ae} is not a directory')
        sys.exit(-1)

    if not (path_to_data.is_file()):
        print(f'{path_to_data} is not a file')
        sys.exit(-1)

    # data quality control
    geno: DataFrame = get_filtered_data(pd.read_csv(path_to_data, index_col=0), path_to_save_qc)
    # normalize the data
    geno: np.ndarray = np.array(get_normalized_data(data=geno))
    minmax_scale(X=geno, feature_range=(0, 1), axis=0, copy=False)
    # setup of training and testing
    geno_train, geno_test = train_test_split(geno, test_size=0.1, random_state=42)

    geno_train_set: GPDataSet = GPDataSet(geno)
    geno_train_set_loader: DataLoader[Any] = DataLoader(dataset=geno_train_set, batch_size=batch_size,
                                                        shuffle=False, num_workers=8)

    geno_test_set = GPDataSet(geno_test)
    geno_test_set_loader: DataLoader[Any] = DataLoader(dataset=geno_test_set, batch_size=batch_size,
                                                       shuffle=False, num_workers=8)
    input_features = len(geno[0])
    output_features = input_features
    smallest_layer = math.ceil(input_features / compression_ratio)
    hidden_layer = int(2 * smallest_layer)

    model = AutoGenoShallow(input_features=input_features, hidden_layer=hidden_layer,
                            smallest_layer=smallest_layer, output_features=output_features).cuda()
    distance = nn.MSELoss()  # for regression, 0, 0.5, 1
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    run_ae(model_name=model_name, model=model, geno_train_set_loader=geno_train_set_loader,
           geno_test_set_loader=geno_test_set_loader, window_size=size, features=input_features,
           optimizer=optimizer, distance=distance, do_train=True, do_test=True, save_dir=path_to_save_ae)


def run_ae(model_name: str, model: AutoGenoShallow, geno_train_set_loader: DataLoader, geno_test_set_loader: DataLoader,
           features: int, optimizer: Adam, distance=nn.MSELoss(), window_size=20, do_train=True,
           do_test=True, save_dir: Path = Path('./model')):
    create_dir(Path(save_dir))
    epoch: int = 0
    test_loss_list: Series = Series([], dtype=float)
    while epoch < 2000:
        input_list: Union[ndarray, int] = np.empty((0, features), dtype=float)
        output_list: Union[ndarray, int] = np.empty((0, features), dtype=float)
        sum_loss: float = 0.0
        r2: float = 0.0
        ks_test: Tuple[float, float] = (0.0, 0.0)
        rho: ndarray = np.asarray([])
        spearman: Tuple[any, float] = (0.0, 0.0)
        if do_train:
            output_coder_list = []
            model.train()
            for geno_data in geno_train_set_loader:
                train_geno: object = Variable(geno_data).float().cuda()
                # =======forward========
                output, coder = model.forward(train_geno)
                loss = distance(output, train_geno)
                sum_loss += loss.item()
                # ======get coder======
                coder2 = coder.cpu().detach().numpy()
                output_coder_list.extend(coder2)
                # ======precision======
                # batch_average_precision = r2_score(y_true=geno_data.cpu().detach().numpy(),
                #                                   y_pred=output.cpu().detach().numpy())
                input_list = np.append(input_list, geno_data.numpy())
                output_list = np.append(output_list, output.cpu().detach().numpy())
                # ======backward========
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # ===========log============
            coder_np: Union[np.ndarray, int] = np.array(output_coder_list)
            coder_file = save_dir.joinpath(f"{model_name}-{str(epoch)}.csv")
            np.savetxt(fname=coder_file, X=coder_np, fmt='%f', delimiter=',')
            # ======goodness of fit======
            ks_test = kstest(rvs=output_list, cdf=input_list)
            spearman: Union[tuple, any] = spearmanr(a=input_list, b=output_list)
            rho = spearman[0]
            r2 = r2_value(y_true=input_list, y_pred=output_list)
        # ===========test==========
        test_input_list: Union[ndarray, int] = np.empty((0, features), dtype=float)
        test_output_list: Union[ndarray, int] = np.empty((0, features), dtype=float)
        test_sum_loss: float = 0.0
        test_r2: float = 0.0
        test_ks_test: Tuple[float, float] = (0.0, 0.0)
        test_rho: ndarray = np.asarray([])
        test_spearman: Tuple[any, float] = (0.0, 0.0)
        if do_test:
            model.eval()
            for geno_test_data in geno_test_set_loader:
                test_geno = Variable(geno_test_data).float().cuda()
                # =======forward========
                test_output, coder = model.forward(test_geno)
                loss = distance(test_output, test_geno)
                test_sum_loss += loss.item()
                # ======precision======
                # batch_average_precision = r2_score(y_true=geno_test_data.cpu().detach().numpy(),
                #                                   y_pred=test_output.cpu().detach().numpy())
                # batch_average_precision = np.mean(r2_value(y_true=geno_test_data.cpu().detach().numpy(),
                #                                           y_pred=test_output.cpu().detach().numpy()))
                test_input_list = np.append(test_input_list, geno_test_data.cpu().detach().numpy())
                test_output_list = np.append(test_output_list, test_output.cpu().detach().numpy())
            # ======goodness of fit======
            test_ks_test = kstest(rvs=test_output_list, cdf=test_input_list)
            test_r2 = r2_value(y_true=test_input_list, y_pred=test_output_list)
            test_loss_list.append(Series([test_sum_loss]))
            test_spearman = spearmanr(a=test_input_list, b=test_output_list)
            test_rho = test_spearman[0]
        print(f"epoch[{epoch + 1:4d}], "
              f"loss: {sum_loss:.4f}, ks: {ks_test[0]:.4f}, p-value: {ks_test[1]:.4f}"
              f", rho: {spearman[0]:.4f}, r2: {r2:.4f}\n        "
              f"test loss: {test_sum_loss:.4f}, ks: {test_ks_test[0]:.4f}, p-value: {test_ks_test[1]:.4f}"
              f", rho: {spearman[0]:.4f}, "
              f"r2: {test_r2:.4f}")
        epoch += 1
        tmp = test_loss_list[-window_size:]
        if round(tmp.mean(), 4) == np.round(test_sum_loss, 4) or \
                (test_loss_list.min() < test_sum_loss):
            break


if __name__ == '__main__':
    if len(sys.argv) < 7:
        print('Default setting are used. Either change AutoEncoder.py to change settings or type:\n')
        print('python AutoEncoder.py model_name original_datafile '
              'quality_control_filename dir_AE_model compression_ratio epoch batch_size')
        print('\tmodel_name - model name e.g. AE_Geno')
        print('\toriginal_datafile - original datafile e.g. ../data_example.csv')
        print('\tquality_control_filename - filename of original data after quality control e.g. ./data_QC.csv')
        print('\tdir_AE_model - base dir to saved AE models e.g. ./AE')
        print('\tcompression_ratio - compression ratio for smallest layer NB: ideally a number that is power of 2')
        print('\twindow_size - window_size for moving average e.g. 20')
        print('\tbatch_size - the size of each batch e.g. 4096')

        main('AE_Geno', Path('../data_example.csv'), Path('./data_QC.csv'), Path('./AE'), 1024, 200, 4096)
    else:
        main(model_name=sys.argv[1], path_to_data=Path(sys.argv[2]), path_to_save_qc=Path(sys.argv[3]),
             path_to_save_ae=Path(sys.argv[4]),
             compression_ratio=int(sys.argv[5]), size=int(sys.argv[6]), batch_size=int(sys.argv[7]))
