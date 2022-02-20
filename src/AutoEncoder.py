# ** Use Python to run Deep Autoencoder (feature selection) **
# ** path - is a string to desired path location. **
import math
import sys
from itertools import islice
from typing import Union, Any, Tuple, Iterable
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from scipy.stats import spearmanr, pearsonr, anderson, anderson_ksamp, levene, ks_2samp
from numpy import ndarray
from pandas import Series, DataFrame
from sklearn.preprocessing import minmax_scale
from torch import nn
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchmetrics import R2Score, SpearmanCorrcoef, PearsonCorrcoef
from sklearn.model_selection import train_test_split
from AutoEncoderModule import GPDataSet, AutoGenoShallow
from CommonTools import create_dir, get_normalized_data
from AutoEncoderModule import get_device, cyclical_lr, get_min_max_lr


def same_distribution_test(*samples: ndarray) -> Tuple[bool, float, float]:
    stat: float
    crit: Union[ndarray, Iterable, int, float]
    stat, crit, _ = anderson_ksamp(samples=samples)

    if crit[2] < stat:
        return False, stat, crit[2]
    else:
        return True, stat, crit[2]


def normality_test(data: ndarray) -> Tuple[bool, float, float]:
    stat: float
    crit: Iterable
    stat, crit, _ = anderson(x=data, dist='norm')
    tmp = next(islice(crit, 2, 3))
    if tmp < stat:
        return False, stat, tmp
    else:
        return True, stat, tmp


def equality_of_variance_test(*samples: ndarray) -> Tuple[bool, float, float]:
    p_value: float
    stat, p_value = levene(samples, center='mean')
    if p_value < 0.5:
        return False, stat, p_value
    else:
        return True, stat, p_value


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
                            smallest_layer=smallest_layer, output_features=output_features).to(get_device())
    distance = nn.MSELoss()  # for regression, 0, 0.5, 1
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.)
    step_size = 4 * len(geno_train_set_loader)
    min_lf , max_lf = get_min_max_lr(model, geno_train_set_loader)
    clr = cyclical_lr(step_size, min_lr=min_lf, max_lr=max_lf)
    scheduler: LambdaLR = torch.optim.lr_scheduler.LambdaLR(optimizer, clr)

    run_ae(model_name=model_name, model=model, geno_train_set_loader=geno_train_set_loader,
           geno_test_set_loader=geno_test_set_loader, window_size=size, features=input_features,
           optimizer=optimizer, distance=distance, do_train=True, do_test=True, save_dir=path_to_save_ae)


def train_model(model, criterion, optimizer, num_epochs=100000, lr_step=False, scheduler=None):
    logs = []

    r2score: R2Score = torchmetrics.R2Score()
    test_r2score: R2Score = torchmetrics.R2Score()
    spearman: SpearmanCorrcoef = torchmetrics.SpearmanCorrcoef()
    test_spearman: SpearmanCorrcoef = torchmetrics.SpearmanCorrcoef()
    pearson: PearsonCorrcoef = torchmetrics.PearsonCorrcoef()
    test_pearson: PearsonCorrcoef = torchmetrics.PearsonCorrcoef()
    for epoch in range(num_epochs):

        epoch_log = {}

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        if lr_step:
                            scheduler.step()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            epoch_log["acc_{}".format(phase)] = epoch_acc
            epoch_log["loss_{}".format(phase)] = epoch_loss

        logs.append(epoch_log)

    return logs


def run_ae(model_name: str, model: AutoGenoShallow, geno_train_set_loader: DataLoader, geno_test_set_loader: DataLoader,
           features: int, optimizer: Adam, distance=nn.MSELoss(), window_size=20, do_train=True,
           do_test=True, save_dir: Path = Path('./model')):
    create_dir(Path(save_dir))
    epoch: int = 0
    test_loss_list: ndarray = np.empty(0, dtype=float)
    while epoch < 9999:
        input_list: Union[ndarray, int] = np.empty((0, features), dtype=float)
        output_list: Union[ndarray, int] = np.empty((0, features), dtype=float)
        sum_loss: float = 0.0
        r2: float = 0.0
        ks_test: Tuple[float, float] = (0.0, 0.0)
        pearson: Tuple[float, any] = (0.0, 0.0)
        spearman: Tuple[any, float] = (0.0, 0.0)
        if do_train:
            output_coder_list = []
            model.train()
            for geno_data in geno_train_set_loader:
                train_geno: object = Variable(geno_data).float().to(get_device())
                # =======forward========
                output, coder = model.forward(train_geno)
                loss = distance(output, train_geno)
                sum_loss += loss.item()
                # ======get coder======
                coder2 = coder.cpu().detach().numpy()
                output_coder_list.extend(coder2)
                # ======goodness of fit======
                # batch_average_precision = r2_score(y_true=geno_data.cpu().detach().numpy(),
                #                                   y_pred=output.cpu().detach().numpy())
                input_list = np.append(input_list, geno_data.numpy())
                output_list = np.append(output_list, output.cpu().detach().numpy())
                # ======backward========
                optimizer.zero_grad()  # clear the gradient
                loss.backward()  # backwork propagation
                # Clip the gradients norm to avoid them becoming too large
                clip_grad_norm_(model.parameters(), 5)

                # Update the LR
                scheduler.step()
                optimizer.step()  # updating gradients
            # ===========log============
            coder_np: Union[np.ndarray, int] = np.array(output_coder_list)
            coder_file = save_dir.joinpath(f"{model_name}-{str(epoch)}.csv")
            np.savetxt(fname=coder_file, X=coder_np, fmt='%f', delimiter=',')
            # ======goodness of fit======
            ks_test = ks_2samp(data1=input_list, data2=output_list)
            spearman = spearmanr(a=input_list, b=output_list)
            pearson = pearsonr(x=input_list, y=output_list)
            r2 = r2_value(y_true=input_list, y_pred=output_list)
        # ===========test==========
        test_input_list: Union[ndarray, int] = np.empty((0, features), dtype=float)
        test_output_list: Union[ndarray, int] = np.empty((0, features), dtype=float)
        test_ks_test: Tuple[float, float] = (0.0, 0.0)
        test_pearson: Tuple[float, any] = (0.0, 0.0)
        test_spearman: Tuple[any, float] = (0.0, 0.0)
        test_sum_loss: float = 0.0
        test_r2: float = 0.0
        if do_test:
            model.eval()
            for geno_test_data in geno_test_set_loader:
                test_geno = Variable(geno_test_data).float().to(get_device())
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
            test_ks_test = ks_2samp(data1=test_input_list, data2=test_output_list)
            test_r2 = r2_value(y_true=test_input_list, y_pred=test_output_list)
            test_loss_list = np.append(test_loss_list, [test_sum_loss])
            test_pearson = pearsonr(x=test_input_list, y=test_output_list)
            test_spearman = spearmanr(a=test_input_list, b=test_output_list)
        if same_distribution_test(input_list, output_list, test_input_list, test_output_list)[0] and \
                normality_test(data=input_list)[0] and \
                equality_of_variance_test(input_list, output_list, test_input_list, test_output_list)[0]:
            print(f"epoch[{epoch + 1:4d}], "
                  f"loss: {sum_loss:.4f}, Pearson: {pearson[0]:.4f}, r2: {r2:.4f}, "
                  f"test loss: {test_sum_loss:.4f}, Pearson: {test_pearson[0]:.4f}, r2: {test_r2:.4f}")
        else:
            print(f"epoch[{epoch + 1:4d}], loss: {sum_loss:.4f}, rho: {spearman[0]:.4f}, r2: {r2:.4f}, "
                  f"test loss: {test_sum_loss:.4f}, rho: {test_spearman[0]:.4f}, r2: {test_r2:.4f}")

        epoch += 1
        tmp: ndarray = test_loss_list[-window_size:]
        if np.round(rolling_mean(x=tmp, size=window_size), 6) == np.round(test_sum_loss, 6) or \
                (test_loss_list.min(initial=10000) < test_sum_loss):
            print(f"\nepoch[{epoch:4d}], "
                  f"loss: {sum_loss:.4f}, ks: {ks_test[0]:.4f}, p-value: {ks_test[1]:.4f}"
                  f", rho: {spearman[0]:.4f}, r2: {r2:.4f}\n        "
                  f"test loss: {test_sum_loss:.4f}, ks: {test_ks_test[0]:.4f}, p-value: {test_ks_test[1]:.4f}"
                  f", rho: {spearman[0]:.4f}, "
                  f"r2: {test_r2:.4f}")
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
