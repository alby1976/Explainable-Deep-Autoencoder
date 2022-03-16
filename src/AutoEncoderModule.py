# python system library
# import argparse
import argparse
import math
import sys
from pathlib import Path
from typing import Any, Union, Dict, Optional, Tuple

# 3rd party modules
import numpy
import numpy as np
import pandas as pd
import pl_bolts.datamodules
import pytorch_lightning as pl
import torch
import torchmetrics as tm
from pandas import DataFrame
from pl_bolts.datamodules import SklearnDataset
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from sklearn.utils import shuffle as sk_shuffle
from torch import nn, Tensor
from torch.nn import functional as f
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DataLoader, TensorDataset

# custom modules
from CommonTools import get_dict_values_1d, get_dict_values_2d, get_transformed_data, DataNormalization, get_data, \
    filter_data


class GPDataModule(pl_bolts.datamodules.SklearnDataModule):
    def __init__(self, x: DataFrame, val_split: float, test_split: float,
                 num_workers: int, random_state: int, shuffle: bool, batch_size: int,
                 pin_memory: bool, drop_last: bool):
        super().__init__(
            x,
            np.zeros(shape=len(x.index)),
            None,
            None,
            None,
            None,
            val_split,
            test_split,
            num_workers,
            random_state,
            shuffle,
            batch_size,
            pin_memory,
            drop_last
        )
        dm = DataNormalization(column_names=x.columns)
        self.train_dataset = SklearnDataset(dm.fit_transform(self.train_dataset.X).to_numpy(), self.train_dataset.Y)
        self.val_dataset = SklearnDataset(dm.transform(self.val_dataset.X).to_numpy(), self.val_dataset.Y)
        self.test_dataset = SklearnDataset(dm.transform(self.test_dataset.X).to_numpy(), self.test_dataset.Y)
        self.size: int = len(x.columns)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        print(f'train: batch size: {self.batch_size} shuffle: {self.shuffle} '
              f'num_workers: {self.num_workers} drop_last: {self.drop_last} pin_memory: {self.pin_memory}')
        loader = DataLoader(
            self.train_dataset.X,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
        return loader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        print(f'val: batch size: {self.batch_size} shuffle: {self.shuffle} '
              f'num_workers: {self.num_workers} drop_last: {self.drop_last} pin_memory: {self.pin_memory}')
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
        return loader

    def test_dataloader(self) -> EVAL_DATALOADERS:
        print(f'test: batch size: {self.batch_size} shuffle: {self.shuffle} '
              f'num_workers: {self.num_workers} drop_last: {self.drop_last} pin_memory: {self.pin_memory}')
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
        return loader

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def teardown(self, stage: Optional[str] = None):
        # Used to clean-up when the run is finished
        ...


class AutoGenoShallow(pl.LightningModule):
    parametric: bool
    testing_dataset: Union[TensorDataset, None]
    train_dataset: Union[TensorDataset, None]

    def __init__(self, save_dir: Path, name: str, ratio: int, cyclical_lr: bool, learning_rate: float,
                 data: Path, transformed_data: Path,
                 batch_size: int, val_split: float, test_split: float,
                 filter_str: str, num_workers: int, random_state: int, shuffle: bool, drop_last: bool,
                 pin_memory: bool):
        super().__init__()  # I guess this inherits __init__ from super class
        # self.testing_dataset = None
        # self.train_dataset = None
        # self.test_input_list = None
        # self.input_list = None

        self.cyclical = cyclical_lr

        # get normalized data quality control
        self.dataset = GPDataModule(
            get_data(geno=filter_data(pd.read_csv(data, index_col=0), filter_str), path_to_save_qc=transformed_data),
            val_split, test_split, num_workers, random_state, shuffle, batch_size, pin_memory, drop_last
        )
        # self.geno: ndarray = get_filtered_data(pd.read_csv(path_to_data, index_col=0), path_to_save_qc).to_numpy()
        self.input_features = self.dataset.size
        self.output_features = self.input_features
        self.smallest_layer = math.ceil(self.input_features / ratio)

        print(f'{self.dataset.size} input_features: {self.input_features} smallest_layer: {self.smallest_layer} ')
        '''
        if tmp < 5:
            self.smallest_layer = 5 if tmp < 5 else tmp
        else:
            self.smallest_layer = tmp
        '''
        self.hidden_layer = 2 * self.smallest_layer
        self.training_r2score_node = tm.R2Score(num_outputs=self.input_features,
                                                multioutput='variance_weighted', compute_on_step=False)
        self.testing_r2score_node = tm.R2Score(num_outputs=self.input_features,
                                               multioutput='variance_weighted', compute_on_step=False)
        self.save_dir = save_dir
        self.model_name = name

        # Hyper-parameters
        self.learning_rate = learning_rate
        self.hparams.batch_size = batch_size
        self.min_lr = self.learning_rate / 6.0
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
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        y: Tensor = self.encoder(x)
        x = self.decoder(y)
        return x, y

    # define training step
    def training_step(self, batch, batch_idx) -> Dict[str, Tensor]:
        output: Tensor
        coder: Tensor
        x: Tensor = batch[0]
        output, coder = self.forward(x)
        self.training_r2score_node.update(preds=output, target=x)
        loss: Tensor = f.mse_loss(input=output, target=x)
        # return {'model': coder, 'loss': loss, 'r2_node': r2_node, 'input': x, 'output': output}
        return {'model': coder.detach(), 'loss': loss}

    # end of training epoch
    def training_epoch_end(self, training_step_outputs):
        # scheduler: CyclicLR = self.lr_schedulers()
        epoch = self.current_epoch

        # extracting training batch data
        losses: Tensor = get_dict_values_1d('loss', training_step_outputs)
        coder: Tensor = get_dict_values_2d('model', training_step_outputs)

        # ===========save model============
        coder_file = self.save_dir.joinpath(f"{self.model_name}-{epoch}.pt")
        torch.save(coder, coder_file)
        # print(f'\n*** save_dir: {self.save_dir} coder_file: {coder_file} ***\n')
        # save_tensor(x=coder, file=coder_file)
        # np.savetxt(fname=coder_file, X=coder_np, fmt='%f', delimiter=',')

        # ======goodness of fit======
        r2_node = self.training_r2score_node.compute()
        print(f"epoch[{epoch + 1:4d}]  "
              f"learning_rate: {self.learning_rate:.6f} "
              f"loss: {losses.sum():.6f}  "
              f"r2_mode: {r2_node}",
              end=' ', file=sys.stderr)

        # logging metrics into log file
        self.log('learning_rate', self.learning_rate, on_step=False, on_epoch=True)
        self.log('loss', torch.sum(losses), on_step=False, on_epoch=True)
        self.log('r2score', r2_node, on_step=False, on_epoch=True)

        # clean up
        self.training_r2score_node.reset()

    # define validation step
    def validation_step(self, batch, batch_idx) -> Dict[str, Tensor]:
        x = batch[0]
        output, _ = self.forward(x)

        # r2_node = self.testing_r2score_node.forward(preds=output, target=x)
        loss = f.mse_loss(input=output, target=x)
        '''
        print(f'{batch_idx} val step batch size: {self.hparams.batch_size} output dim: {output.size()} '
              f'batch dim: {x.size()} loss dim: {loss.size()}')
        '''
        # return {'loss': loss, 'r2_node': r2_node, 'input': x, 'output': output}
        return {'loss': loss}

    # end of validation epoch
    def validation_epoch_end(self, testing_step_outputs):
        # extracting training batch data
        losses = get_dict_values_1d('loss', testing_step_outputs)
        # print(f'regular losses: {losses.size()} pred: {pred.size()} target: {target.size()}')

        # ======goodness of fit ======
        r2_node: Tensor = self.testing_r2score_node.compute()
        print(f"test_loss: {torch.sum(losses):.6f} "
              f"test_r2_node: {r2_node}",
              file=sys.stderr)

        # logging validation metrics into log file
        self.log('testing_loss', torch.sum(losses), on_step=False, on_epoch=True)
        self.log('testing_r2score', r2_node, on_step=False, on_epoch=True)
        self.testing_r2score_node.reset()

    # configures the optimizers through learning rate
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        if self.cyclical:
            it_per_epoch = math.ceil(len(self.train_dataloader()) / self.hparams.batch_size)
            print(f'it_per_epoch: {it_per_epoch}')
            scheduler: CyclicLR = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=self.min_lr,
                                                                    mode='exp_range',
                                                                    cycle_momentum=False,
                                                                    step_size_up=4 * it_per_epoch,
                                                                    max_lr=self.learning_rate)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

        return {"optimizer": optimizer}

    def setup(self, stage: Optional[str] = None):
        # setup of training and testing
        pass

    def train_dataloader(self) -> EVAL_DATALOADERS:
        # Called when training the model
        self.dataset.batch_size = self.hparams.batch_size
        # print(f'input_list: {type(self.input_list)} train_data: {type(self.train_dataset)}')
        return self.dataset.train_dataloader()

    def val_dataloader(self) -> EVAL_DATALOADERS:
        # Called when evaluating the model (for each "n" steps or "n" epochs)
        self.dataset.batch_size = self.hparams.batch_size
        # print(f'test_input_list: {type(self.test_input_list)} testing_data: {type(self.testing_dataset)}')
        return self.dataset.val_dataloader()

    def test_dataloader(self) -> EVAL_DATALOADERS:
        self.dataset.batch_size = self.hparams.batch_size
        return self.dataset.test_dataloader()

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        self.dataset.batch_size = self.hparams.batch_size
        return self.dataset.predict_dataloader()

    def _forward_unimplemented(self, *inputs: Any) -> None:
        pass

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser):
        parser = parent_parser.add_argument_group("AutoGenoShallow")
        # parser.add_argument("--data", type=str, default='../data_example.csv',
        #                     help='original datafile e.g. ../data_example.csv')
        # parser.add_argument("--transformed_data", type=str, default="./data_QC.csv",
        #                     help='filename of original data after quality control e.g. ./data_QC.csv')
        # parser.add_argument("--batch_size", type=int, default=64, help='the size of each batch e.g. 64')

        parser.add_argument("-n", "--name", type=str, default='AE_Geno',
                            help='model name e.g. AE_Geno')
        parser.add_argument("-sd", "--save_dir", type=Path,
                            default=Path(__file__).absolute().parent.parent.joinpath("AE"),
                            help='base dir to saved AE models e.g. ./AE')
        parser.add_argument("-cr", "--ratio", type=int, default=8,
                            help='compression ratio for smallest layer NB: ideally a number that is power of 2')
        parser.add_argument("-lr", "--learning_rate", type=float, default=0.0001,
                            help='the base learning rate for training e.g 0.0001')
        parser.add_argument("--data", type=Path,
                            default=Path(__file__).absolute().parent.parent.joinpath("data_example.csv"),
                            help='original datafile e.g. ./data_example.csv')
        parser.add_argument("-td", "--transformed_data", type=Path,
                            default=Path(__file__).absolute().parent.parent.joinpath("data_QC.csv"),
                            help='filename of original data after quality control e.g. ./data_QC.csv')
        parser.add_argument("-bs", "--batch_size", type=int, default=64, help='the size of each batch e.g. 64')
        parser.add_argument("-vs", "--val_split", type=float, default=0.1,
                            help='validation set split ratio. default is 0.1')
        parser.add_argument("-ts", "--test_split", type=float, default=0.0,
                            help='test set split ratio. default is 0.0')
        parser.add_argument("-w", "--num_workers", type=int, default=0,
                            help='number of processors used to load data. ie worker = 4 * # of GPU. default is 0')
        parser.add_argument("-f", "--filter_str", type=str, default="",
                            help='filter string to select which rows are processed. default: \'\'')
        parser.add_argument("-rs", "--random_state", type=int, default=42,
                            help='sets a seed to the random generator, so that your train-val-test splits are '
                                 'always deterministic. default is 42')
        parser.add_argument("-s", "--shuffle", type=bool, default=False,
                            help='whether to shuffle the dataset before splitting the dataset. default is False.')
        parser.add_argument("-c", "--cyclical_lr", type=bool, default=False,
                            help='whether to use cyclical learning rate or not. default is False.')
        parser.add_argument("--drop_last", type=bool, default=False,
                            help='whether to drop the last column or not. default is False.')
        parser.add_argument("--pin_memory", type=bool, default=True,
                            help='whether to pin_memory or not. default is True.')
        return parent_parser
