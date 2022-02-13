# ** Use Python to run Deep Autoencoder (feature selection) **
# ** path - is a string to desired path location. **
import math
import sys
from typing import Union
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from numpy import ndarray
from pandas import Series, DataFrame
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import r2_score
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from AutoEncoderModule import GPDataSet
from AutoEncoderModule import AutoGenoShallow
from AutoEncoderModule import create_dir
from AutoEncoderModule import get_normalized_data


def r2_value(y_true: ndarray, y_pred: ndarray) -> ndarray:
    # num: int = features*(np.sum(y_true*y_pred) - (y_true.sum()*y_pred.sum()))
    # den: int = np.sqrt((features * np.sum(np.square(y_true)) - np.square(y_true.sum())) *
    #                   (features * np.sum(np.square(y_pred)) - np.square(y_pred.sum())))
    y_ave = y_true.mean(axis=0)
    ssr: ndarray = np.square(y_pred - y_ave)
    sst: ndarray = np.square(y_true - y_ave)
    '''
    print(f'y_ave {y_ave.shape}: {y_ave}')
    print(f'y_true {y_true.shape}:\n{y_true}')
    print(f'y_pred {y_pred.shape}:\n{y_pred}')
    print(f'ssr {ssr.shape}:\n{ssr}')
    print(f'sst {ssr.shape}:\n{sst}')
    print(f'ssr: {ssr.sum(axis=0)}\nsst:\n{sst.sum(axis=0)}\nr^2:\n{ssr.sum(axis=0)/sst.sum(axis=0)}')
    sys.exit(-1)
    '''
    return ssr.sum(axis=0) / sst.sum(axis=0)


def get_filtered_data(geno: DataFrame, path_to_save_qc: Path) -> DataFrame:
    geno_var: Union[Series, int] = geno.var()
    geno.drop(geno_var[geno_var < 1].index.values, axis=1, inplace=True)
    create_dir(path_to_save_qc.parent)
    geno.to_csv(path_to_save_qc)
    return geno


def run_ae(model_name: str, model: AutoGenoShallow, geno_train_set_loader: DataLoader, geno_test_set_loader: DataLoader,
           features: int, optimizer: Adam, distance=nn.MSELoss(), num_epochs=200, do_train=True,
           do_test=True, save_dir: Path = Path('./model')):
    create_dir(Path(save_dir))
    for epoch in range(num_epochs):
        batch_precision_list = []
        output_coder_list = []
        average_precision = 0.0
        sum_loss = 0.0
        if do_train:
            input_list: ndarray = np.empty((0, features), float)
            output_list: ndarray = np.empty((0, features), float)
            current_batch: int = 0
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
                # batch_average_precision = r2_score(y_true=geno_data.cpu().detach().numpy(),
                #                                   y_pred=output.cpu().detach().numpy())
                y_true = np.asarray([1 if x >= 0.5 else 0 for x in geno_data.numpy()])
                y_pred = np.asarray([1 if x >= 0.5 else 0 for x in output.cpu().detach().numpy()])
                tp = np.count_nonzero(y_true)
                fp = np.count_nonzero(np.asarray([1*(bool(y) and not(bool(x))) for x, y in zip(y_true, y_pred)]))

                rows, columns = true.shape
                batch_average_precision = tp / (tp + fp)

                # batch_average_precision = np.mean(r2_value(y_true=geno_data.cpu().detach().numpy(),
                #                                           y_pred=output.cpu().detach().numpy()))
                # print(f'batch: {current_batch} r2 value: {batch_average_precision}')
                batch_precision_list.append(batch_average_precision)
                # input_list = np.append(input_list, geno_data.cpu().detach().numpy(), axis=0)
                # output_list = np.append(output_list, output.cpu().detach().numpy(), axis=0)

                # ======backward========
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # ===========log============
            coder_np: Union[np.ndarray, int] = np.array(output_coder_list)
            coder_file = save_dir.joinpath(f"{model_name}-{str(epoch)}.csv")
            np.savetxt(fname=coder_file, X=coder_np, fmt='%f', delimiter=',')
            # batch_precision_list = [r2_score_batch1, r2_score_batch2,...]
            average_precision = np.mean(np.asarray(batch_precision_list))
        # ===========test==========
        test_batch_precision_list = []
        test_average_precision = 0.0
        test_sum_loss = 0.0
        if do_test:
            test_input_list: ndarray = np.empty((0, features), float)
            test_output_list: ndarray = np.empty((0, features), float)
            test_current_batch: int = 0
            model.eval()
            for geno_test_data in geno_test_set_loader:
                test_current_batch += 1
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
                true = geno_test_data.numpy()
                pred = test_output.cpu().detach().numpy()

                rows, columns = true.shape
                batch_average_precision = 1 - np.count_nonzero(true - pred) / (rows * columns)
                test_batch_precision_list.append(batch_average_precision)
                # test_input_list = np.append(test_input_list, geno_test_data.cpu().detach().numpy(), axis=0)
                # test_output_list = np.append(test_output_list, test_output.cpu().detach().numpy(), axis=0)
            # test_batch_precision_list = [r2_score_batch1, r2_score_batch2,...]
            test_average_precision = np.mean(np.asarray(test_batch_precision_list))
        print(f"epoch[{epoch + 1:3d}/{num_epochs}, loss: {sum_loss:.4f}, precision: {average_precision:.4f}, "
              f" test lost: {test_sum_loss:.4f}, test precision: {test_average_precision:.4f}")


def main(model_name: str, path_to_data: Path, path_to_save_qc: Path, path_to_save_ae: Path,
         compression_ratio: int, epoch: int, batch_size: int):
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
    geno_train_set_loader = DataLoader(dataset=geno_train_set, batch_size=batch_size, shuffle=False, num_workers=8)

    geno_test_set = GPDataSet(geno_test)
    geno_test_set_loader = DataLoader(dataset=geno_test_set, batch_size=batch_size, shuffle=False, num_workers=8)
    input_features = len(geno[0])
    output_features = input_features
    smallest_layer = math.ceil(input_features / compression_ratio)
    hidden_layer = int(2 * smallest_layer)

    model = AutoGenoShallow(input_features=input_features, hidden_layer=hidden_layer,
                            smallest_layer=smallest_layer, output_features=output_features).cuda()
    distance = nn.MSELoss()  # for regression, 0, 0.5, 1
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    run_ae(model_name=model_name, model=model, geno_train_set_loader=geno_train_set_loader,
           features=input_features, geno_test_set_loader=geno_test_set_loader, num_epochs=epoch,
           optimizer=optimizer, distance=distance, do_train=True, do_test=True, save_dir=path_to_save_ae)


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
        print('\tepoch - number of iterations e.g. 200')
        print('\tbatch_size - the size of each batch e.g. 4096')

        main('AE_Geno', Path('../data_example.csv'), Path('./data_QC.csv'), Path('./AE'), 1024, 200, 4096)
    else:
        main(model_name=sys.argv[1], path_to_data=Path(sys.argv[2]), path_to_save_qc=Path(sys.argv[3]),
             path_to_save_ae=Path(sys.argv[4]),
             compression_ratio=int(sys.argv[5]), epoch=int(sys.argv[6]), batch_size=int(sys.argv[7]))
