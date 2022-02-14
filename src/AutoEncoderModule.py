from typing import Any, Union
from pathlib import Path
from pandas import DataFrame
from torch import nn
from torch.utils.data import Dataset
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
