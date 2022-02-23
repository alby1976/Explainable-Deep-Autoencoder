import math
from pathlib import Path
from typing import Any, Union, Dict, Optional

import torch
import torchmetrics
from numpy import ndarray
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from sklearn.model_selection import train_test_split
from torch import nn, Tensor
from torch.nn import functional as f
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import numpy as np
import pandas as pd

from CommonTools import data_parametric, get_dict_values, get_data


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


class AutoGenoShallow(pl.LightningModule):
    def __init__(self, save_dir: Path, path_to_data: Path, path_to_save_qc: Path,
                 model_name: str, compression_ratio: int, batch_size: int = 32,
                 learning_rate: float = 0.0001):
        super().__init__()  # I guess this inherits __init__ from super class
        self.testing_dataset = None
        self.train_dataset = None
        self.test_input_list = None
        self.input_list = None
        self.geno = None
        self.input_features = 1
        self.output_features = 1
        self.smallest_layer = 1
        self.hidden_layer = 2
        self.training_r2score = torchmetrics.R2Score(compute_on_step=False)
        self.training_pearson = torchmetrics.PearsonCorrcoef(compute_on_step=False)
        self.training_spearman = torchmetrics.SpearmanCorrcoef(compute_on_step=False)
        self.testing_r2score = torchmetrics.R2Score(compute_on_step=False)
        self.testing_pearson = torchmetrics.PearsonCorrcoef(compute_on_step=False)
        self.testing_spearman = torchmetrics.SpearmanCorrcoef(compute_on_step=False)
        self.save_dir = save_dir
        self.path_to_data = path_to_data
        self.path_to_save_qc = path_to_save_qc
        self.model_name = model_name
        self.compression_ratio = compression_ratio

        # Hyper-parameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.min_lf = learning_rate / 6.0
        self.save_hyperparameters()

        # def the encoder function
        self.encoder = nn.Sequential(
            nn.Linear(self.input_features, self.hidden_layer),
            nn.ReLU(True),
            nn.Linear(self.hidden_layer, self.smallest_layer),
            nn.ReLU(True),
        )

        # def the decoder function
        self.decoder = nn.Sequential(
            nn.Linear(self.smallest_layer, self.hidden_layer),
            nn.ReLU(True),
            nn.Linear(self.hidden_layer, self.output_features),
            nn.Sigmoid()
        )

    # define forward function
    def forward(self, x):
        y = self.encoder(x)
        x = self.decoder(y)
        return x, y

    # define training step
    def training_step(self, batch, batch_idx) -> Dict[str, Tensor]:
        output, coder = self.forward(batch)
        self.training_r2score(preds=output, target=batch)
        self.training_spearman(preds=output, target=batch)
        self.training_pearson(preds=output, target=batch)
        loss = f.mse_loss(input=output, target=batch)
        return {'input': batch, 'output': output, 'model': coder, 'loss': loss}

    # end of training epoch
    def training_epoch_end(self, training_step_outputs):
        losses: Tensor = get_dict_values('loss', training_step_outputs)
        epoch = self.trainer.current_epoch

        # ===========save model============
        output_coder_list: Tensor = get_dict_values('model', training_step_outputs)
        coder_np: Union[np.ndarray, int] = output_coder_list.cpu().detach().numpy()
        coder_file = self.save_dir.joinpath(f"{self.model_name}-{epoch}.csv")
        np.savetxt(fname=coder_file, X=coder_np, fmt='%f', delimiter=',')

        # ======goodness of fit======
        r2: float = self.training_r2score.compute().item()
        coefficient: float
        if data_parametric:
            coefficient = self.training_pearson.compute().item()
        else:
            coefficient = self.training_spearman.compute().item()

        self.log('step', epoch + 1)
        self.log('loss', losses.sum())
        self.log('r2score', self.training_r2score, on_step=False, on_epoch=True)

        print(f"epoch[{epoch + 1:4d}], "
              f"loss: {losses.sum():.4f}, coefficient: {coefficient:.4f}, r2: {r2:.4f},",
              end=' ')

    # define validation step
    def validation_step(self, batch, batch_idx) -> Dict[str, Tensor]:
        output, coder = self.forward(batch)
        self.training_r2score(preds=output, target=batch)
        self.training_spearman(preds=output, target=batch)
        self.training_pearson(preds=output, target=batch)
        loss = f.mse_loss(input=output, target=batch)
        return {'input': batch, 'output': output, 'model': coder, 'loss': loss}

    # end of validation epoch
    def validation_epoch_end(self, testing_step_outputs):
        losses = np.asarray(get_dict_values('loss', testing_step_outputs))
        # epoch = self.trainer.current_epoch

        # ======goodness of fit======
        r2 = self.testing_r2score.compute().item()
        coefficient: float
        if data_parametric:
            coefficient = self.testing_pearson.compute().item()
        else:
            coefficient = self.testing_spearman.compute().item()

        # self.log('step', epoch + 1)
        self.log('test_loss', losses.sum())
        self.log('test_r2score', self.testing_r2score, on_step=False, on_epoch=True)

        print(f"test_loss: {losses.sum():.4f}, "
              f"test_coefficient: {coefficient:.4f}, test_r2: {r2:.4f}")

    # configures the optimizers through learning rate
    def configures_optimizer(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        step_size = 4 * len(self.train_dataloader())
        clr = self.cyclical_lr(step_size, min_lr=self.min_lf, max_lr=self.learning_rate)
        scheduler: LambdaLR = torch.optim.lr_scheduler.LambdaLR(optimizer, [clr])
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def prepare_data(self) -> None:
        # get normalized data quality control
        self.geno: ndarray = get_data(pd.read_csv(self.path_to_data, index_col=0), self.path_to_save_qc)
        self.input_features = len(self.geno[0])
        self.output_features = self.input_features
        self.smallest_layer = math.ceil(self.input_features / self.compression_ratio)
        self.hidden_layer = int(2 * self.smallest_layer)

    def setup(self, stage: Optional[str] = None):
        # setup of training and testing
        geno_train, geno_test = train_test_split(self.geno, test_size=0.1, random_state=42)
        # Assign train/val datasets for using in data-loaders
        if stage == 'fit' or stage is None:
            self.input_list = torch.from_numpy(geno_train).type(torch.FloatTensor)
            self.test_input_list = torch.from_numpy(geno_test).type(torch.FloatTensor)

    def train_dataloader(self):
        # Called when training the model
        self.train_dataset = torch.utils.data.TensorDataset(self.input_list)
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        # Called when evaluating the model (for each "n" steps or "n" epochs)
        self.testing_dataset = torch.utils.data.TensorDataset(self.test_input_list)
        return DataLoader(self.testing_dataset, batch_size=self.batch_size)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def _forward_unimplemented(self, *inputs: Any) -> None:
        pass

    # get current learning rate
    def get_lr(self) -> float:
        return self.learning_rate

    @staticmethod
    def cyclical_lr(step_size: int, min_lr: float = 3e-2, max_lr: float = 3e-3):
        # Scaler: we can adapt this if we do not want the triangular CLR
        def scaler(x: Any) -> int:
            return x * 0 + 1

        # Additional function to see where on the cycle we are
        def relative(it, size: int):
            cycle = math.floor(1 + it / (2 * size))
            x = abs(it / size - 2 * cycle + 1)
            return max(0, (1 - x)) * scaler(cycle)

        return lambda it: min_lr + (max_lr - min_lr) * relative(it, step_size)  # Lambda function to calculate the LR
