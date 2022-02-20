import math
from typing import Any, Tuple
from pathlib import Path

import torch
from numpy import ndarray
from pandas import DataFrame
from torch import nn, device
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import numpy as np


class GPDataSet(Dataset):
    def __init__(self, gp_list):
        # 'Initialization'
        self.gp_list = gp_list

    def __len__(self):
        # 'Denotes the total number of samples'
        return len(self.gp_list)

    def __getitem__(self, index):
        # 'Generates one sample of data'
        # Load data and get label
        x = self.gp_list[index]
        x = np.array(x)
        return x


class AutoGenoShallow(nn.Module):
    def __init__(self, input_features, hidden_layer, smallest_layer, output_features):
        super().__init__()  # I guess this inherits __init__ from super class

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

    def _forward_unimplemented(self, *inputs: Any) -> None:
        pass

    # def forward function
    def forward(self, x):
        y = self.encoder(x)
        x = self.decoder(y)
        return x, y


def get_normalized_data(data: DataFrame) -> DataFrame:
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    return DataFrame(data=scaler.fit_transform(data), columns=data.columns)


def create_dir(directory: Path):
    """make a directory (directory) if it doesn't exist"""
    directory.mkdir(parents=True, exist_ok=True)


def get_device() -> device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def cyclical_lr(step_size: int, min_lr: float = 3e-2, max_lr: float = 3e-3):
    # Scaler: we can adapt this if we do not want the triangular CLR
    def scaler(x: Any) -> int:
        return 1

    # Additional function to see where on the cycle we are
    def relative(it, size: int):
        cycle = math.floor(1 + it / (2 * size))
        x = abs(it / size - 2 * cycle + 1)
        return max(0, (1 - x)) * scaler(cycle)

    return lambda it: min_lr + (max_lr - min_lr) * relative(it, step_size) # Lambda function to calculate the LR


def get_min_max_lr(model: nn.Module, dataset: DataLoader) -> Tuple[float, float]:
    lr_find_loss: ndarray
    lr_find_lr: ndarray
    lr_find_loss, lr_find_lr = get_lr_parameter_list(model, dataset)
    lr_max = lr_find_lr[lr_find_lr.argmin()] / 10
    lr_min = lr_max / 6.0

    return lr_min, lr_max


def get_lr_parameter_list(model: nn.Module, dataset: DataLoader) -> Tuple[ndarray, ndarray]:
    # Experiment parameters
    lr_find_epochs = 2
    start_lr = 1e-7
    end_lr = 0.1
    # Set up the model, optimizer and loss function for the experiment

    optimizer: Adam = torch.optim.Adam(model.parameters(), start_lr)
    criterion = nn.MSELoss()

    # y = a.e(-bt)
    # end_lr = start_lr . e(b.t)
    # (end_lr - start_lr) = e(b.t)
    # ln(end_lr - start_lr) = b.t
    # b = ln(end_lr - start_lr) / t
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                  lambda x: math.exp(x * math.log(x=end_lr / start_lr, base=math.e) /
                                                                     (lr_find_epochs * len(dataset))))
    # Run the experiment

    lr_find_loss = []
    lr_find_lr = []

    iteration = 0

    smoothing = 0.05

    for i in range(lr_find_epochs):
        print("epoch {}".format(i))
        for data in dataset:

            # Send to device
            training_data = Variable(data).float().to(get_device())

            # Training mode and zero gradients
            model.train()
            optimizer.zero_grad()

            # Get outputs to calc loss
            outputs = model(training_data)
            loss = criterion(outputs, training_data)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Update LR
            scheduler.step()
            lr_step = optimizer.state_dict()["param_groups"][0]["lr"]
            lr_find_lr.append(lr_step)

            # smooth the loss
            if iter == 0:
                lr_find_loss.append(loss)
            else:
                loss = smoothing * loss + (1 - smoothing) * lr_find_loss[-1]
                lr_find_loss.append(loss)

            iteration += 1
    return np.asarray(lr_find_loss), np.asarray(lr_find_lr)
