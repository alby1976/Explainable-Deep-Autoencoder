import math
import sys
from pathlib import Path
from typing import Any, Union, Dict, Optional

import numpy as np
import pandas as pd
from pytorch_lightning import LightningModule
import torch
import torchmetrics
from numpy import ndarray
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from sklearn.model_selection import train_test_split
from torch import nn, Tensor
from torch.nn import functional as f
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import Dataset, DataLoader
from CommonTools import data_parametric, get_dict_values_1d, get_dict_values_2d, get_data, \
    r2_value_weighted, same_distribution_test


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


class AutoGenoShallow(LightningModule):
    def __init__(self, save_dir: Path, path_to_data: Path, path_to_save_qc: Path,
                 model_name: str, compression_ratio: int, batch_size: int = 32,
                 learning_rate: float = 0.0001):
        super().__init__()  # I guess this inherits __init__ from super class
        self.testing_dataset = None
        self.train_dataset = None
        self.test_input_list = None
        self.input_list = None
        # get normalized data quality control
        self.geno: ndarray = get_data(pd.read_csv(path_to_data, index_col=0), path_to_save_qc)
        # self.geno: ndarray = get_filtered_data(pd.read_csv(path_to_data, index_col=0), path_to_save_qc).to_numpy()
        self.input_features = len(self.geno[0])
        self.output_features = self.input_features
        tmp = math.ceil(self.input_features / compression_ratio)
        if tmp < 5:
            self.smallest_layer = 5 if tmp < 5 else tmp
        else:
            self.smallest_layer = tmp
        self.hidden_layer = 2 * self.smallest_layer
        self.training_r2score_node = torchmetrics.R2Score(num_outputs=self.input_features,
                                                          multioutput='raw_values', compute_on_step=False)
        self.training_pearson = torchmetrics.regression.pearson.PearsonCorrcoef(compute_on_step=True)
        self.training_spearman = torchmetrics.regression.spearman.SpearmanCorrcoef(compute_on_step=True)
        self.testing_r2score_node = torchmetrics.R2Score(num_outputs=self.input_features,
                                                         multioutput='raw_values', compute_on_step=False)
        self.testing_pearson = torchmetrics.regression.pearson.PearsonCorrcoef(compute_on_step=True)
        self.testing_spearman = torchmetrics.regression.spearman.SpearmanCorrcoef(compute_on_step=True)
        self.save_dir = save_dir
        self.model_name = model_name

        # Hyper-parameters
        self.learning_rate = learning_rate
        self.hparams.batch_size = batch_size
        self.min_lr = self.learning_rate / 6.0
        self.save_hyperparameters()

        # def the encoder function
        self.encoder = nn.Sequential(
            nn.Linear(self.input_features, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
        )

        # def the decoder function
        self.decoder = nn.Sequential(
            nn.Linear(self.input_features, self.hidden_layer),
            nn.ReLU(True),
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
        x = batch[0]
        # print(f'{batch_idx} training batch size: {self.hparams.batch_size} x: {x.size()}')
        output, coder = self.forward(x)
        r2_node = self.training_r2score_node.forward(preds=output, target=x)
        loss = f.mse_loss(input=output, target=x)
        return {'model': coder, 'loss': loss, 'r2_node': r2_node, 'input': x, 'output': output}

    # end of training epoch
    def training_epoch_end(self, training_step_outputs):
        scheduler: CyclicLR = self.lr_schedulers()
        epoch = self.current_epoch

        # extracting training batch data
        losses: Tensor = get_dict_values_1d('loss', training_step_outputs)
        x: Tensor = get_dict_values_2d('input', training_step_outputs)
        output: Tensor = get_dict_values_2d('output', training_step_outputs)
        numpy_x: ndarray = x.cpu().detach().numpy()
        numpy_output: ndarray = output.cpu().detach().numpy()

        try:
            r2_node: Tensor = get_dict_values_1d('r2_node', training_step_outputs)
        except TypeError:
            r2_node = self.training_r2score_node.compute()

        # ===========save model============
        output_coder_list: Tensor = get_dict_values_2d('model', training_step_outputs)
        coder_np: Union[np.ndarray, int] = output_coder_list.cpu().detach().numpy()
        coder_file = self.save_dir.joinpath(f"{self.model_name}-{epoch}.csv")
        # print(f'\n*** save_dir: {self.save_dir} coder_file: {coder_file} ***\n')
        np.savetxt(fname=coder_file, X=coder_np, fmt='%f', delimiter=',')

        # ======goodness of fit======
        result: ndarray = np.asarray([data_parametric(numpy_x[:, i], numpy_output[:, i])
                                      for i in range(self.input_features)])
        anderson: ndarray = np.asarray([same_distribution_test(numpy_x[:, i], numpy_output[:, i])
                                        for i in range(self.input_features)])

        coefficient: Tensor
        if np.all(result):
            coefficient = torch.stack(
                [self.training_pearson.forward(preds=torch.index_select(output, 1, torch.tensor([i],
                                                                                                device=output.device)),
                                               target=torch.index_select(x, 1, torch.tensor([i],
                                                                                            device=x.device)))
                 for i in range(x.size(dim=1))])
        else:
            coefficient = torch.stack(
                [self.training_spearman.forward(preds=torch.index_select(output, 1, torch.tensor([i],
                                                                                                 device=output.device)),
                                                target=torch.index_select(x, 1, torch.tensor([i],
                                                                                             device=x.device)))
                 for i in range(x.size(dim=1))])

        # print(f'train coefficient: {coefficient.size()}\n{coefficient}')

        print(f"epoch[{epoch + 1:4d}]  learning_rate: {scheduler.get_last_lr()[0]:.6f} "
              f"loss: {losses.sum().item():.4f}  parametric: {np.all(result)} "
              f"coefficient: {torch.mean(coefficient).item():.4f} "
              f"r2_mode: {r2_value_weighted(y_true=x, y_pred=output).item():.4f}",
              end=' ', file=sys.stderr)

        # logging metrics into log file
        self.log('learning_rate', scheduler.get_last_lr()[0], on_step=False, on_epoch=True)
        self.log('train_anderson_darling_test', torch.from_numpy(anderson).type(torch.FloatTensor),
                 on_step=False, on_epoch=True)
        self.log('loss', torch.sum(losses), on_step=False, on_epoch=True)
        self.log('parametric', int(np.all(result)), on_step=False, on_epoch=True)
        self.log('coefficient', torch.mean(coefficient), on_step=False, on_epoch=True)
        self.log('r2score_per_node', r2_value_weighted(y_true=x, y_pred=output), on_step=False, on_epoch=True)
        self.log('r2score_per_node_raw', r2_node, on_step=False, on_epoch=True)

    # define validation step
    def validation_step(self, batch, batch_idx) -> Dict[str, Tensor]:
        x = batch[0]
        output, _ = self.forward(x)

        r2_node = self.testing_r2score_node.forward(preds=output, target=x)
        loss = f.mse_loss(input=output, target=x)
        '''
        print(f'{batch_idx} val step batch size: {self.hparams.batch_size} output dim: {output.size()} '
              f'batch dim: {x.size()} loss dim: {loss.size()}')
        '''
        return {'loss': loss, 'r2_node': r2_node, 'input': x, 'output': output}

    # end of validation epoch
    def validation_epoch_end(self, testing_step_outputs):
        losses = get_dict_values_1d('loss', testing_step_outputs)
        x = get_dict_values_2d('input', testing_step_outputs)
        output = get_dict_values_2d('output', testing_step_outputs)
        np_x: ndarray = x.cpu().detach().numpy()
        np_output: ndarray = output.cpu().detach().numpy()

        try:
            r2_node = get_dict_values_1d('r2_node', testing_step_outputs)
        except TypeError:
            r2_node = self.testing_r2score_node.compute()
        # print(f'regular losses: {losses.size()} pred: {pred.size()} target: {target.size()}')

        # ======goodness of fit======
        anderson: ndarray = np.asarray([same_distribution_test(np_x[:, i], np_output[:, i])
                                        for i in range(self.input_features)])
        result: ndarray = np.asarray([data_parametric(np_x[:, i], np_output[:, i])
                                      for i in range(self.input_features)])

        coefficient: Tensor
        if np.all(result):
            coefficient = torch.stack(
                [self.testing_pearson.forward(preds=torch.index_select(output, 1, torch.tensor([i],
                                                                                               device=output.device)),
                                              target=torch.index_select(x, 1, torch.tensor([i],
                                                                                           device=x.device)))
                 for i in range(x.size(dim=1))])
        else:
            coefficient = torch.stack(
                [self.testing_spearman.forward(preds=torch.index_select(output, 1, torch.tensor([i],
                                                                                                device=output.device)),
                                               target=torch.index_select(x, 1, torch.tensor([i],
                                                                                            device=x.device)))
                 for i in range(x.size(dim=1))])

        print(f"test_loss: {torch.sum(losses).item():.4f} test_parm: {np.all(result)} test_coefficient: "
              f"{torch.mean(coefficient).item():.4f} "
              f"test_r2_node: {r2_value_weighted(y_true=x, y_pred=output).item():.4f}", file=sys.stderr)

        # logging validation metrics into log file
        self.log('testing_loss', torch.sum(losses))
        self.log('testing_anderson_darling_test', torch.from_numpy(anderson).type(torch.FloatTensor),
                 on_step=False, on_epoch=True)
        self.log('testing_parametric', int(np.all(result)), on_step=False, on_epoch=True)
        self.log('testing_coefficient', torch.mean(coefficient), on_step=False, on_epoch=True)
        self.log('testing_r2score_per_node', r2_value_weighted(y_true=x, y_pred=output), on_step=False, on_epoch=True)
        self.log('testing_r2score_per_node_raw', r2_node, on_step=False, on_epoch=True)

    # configures the optimizers through learning rate
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        it_per_epoch = len(self.train_dataset) / self.batch_size
        print(f'it_per_epoch: {it_per_epoch}')
        scheduler: CyclicLR = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=self.min_lr,
                                                                mode='exp_range',
                                                                cycle_momentum=False,
                                                                step_size_up=4 * it_per_epoch,
                                                                max_lr=self.learning_rate)
        # step_size = 4 * len(self.train_dataloader())
        # clr = self.cyclical_lr(step_size, min_lr=self.min_lr, max_lr=self.learning_rate)
        # scheduler: LambdaLR = torch.optim.lr_scheduler.LambdaLR(optimizer, [clr])

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def setup(self, stage: Optional[str] = None):
        # setup of training and testing
        geno_train, geno_test = train_test_split(self.geno, test_size=0.1, random_state=42)
        # print(f'geno_train dim:{geno_train.shape} geno_test dim: {geno_test.shape}')
        # Assign train/val datasets for using in data-loaders
        if stage == 'fit' or stage is None:
            self.input_list = torch.from_numpy(geno_train).type(torch.FloatTensor)
            self.test_input_list = torch.from_numpy(geno_test).type(torch.FloatTensor)

    def train_dataloader(self) -> EVAL_DATALOADERS:
        # Called when training the model
        self.train_dataset = torch.utils.data.TensorDataset(self.input_list)
        # print(f'input_list: {type(self.input_list)} train_data: {type(self.train_dataset)}')
        return DataLoader(dataset=self.train_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=8)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        # Called when evaluating the model (for each "n" steps or "n" epochs)
        self.testing_dataset = torch.utils.data.TensorDataset(self.test_input_list)
        # print(f'test_input_list: {type(self.test_input_list)} testing_data: {type(self.testing_dataset)}')
        return DataLoader(dataset=self.testing_dataset, batch_size=self.hparams.batch_size,
                          shuffle=False, num_workers=8)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def _forward_unimplemented(self, *inputs: Any) -> None:
        pass

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
