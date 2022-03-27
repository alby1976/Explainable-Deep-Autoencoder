# python system library
# import argparse
import argparse
import gc
import math
import sys
from pathlib import Path
from typing import Any, Union, Dict, Optional, Tuple

# 3rd party modules
import numpy as np
import pandas as pd
import pl_bolts.datamodules
import pytorch_lightning as pl
import torch
import torchmetrics as tm
from pandas import DataFrame, Series
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from sklearn.model_selection import train_test_split
from torch import nn, Tensor
from torch.nn import functional as f
from torch.optim.swa_utils import SWALR
from torch.utils.data import DataLoader, TensorDataset, Dataset

# custom modules
from CommonTools import get_dict_values_1d, get_dict_values_2d, DataNormalization, get_data


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


class GPDataModule(pl_bolts.datamodules.SklearnDataModule):
    def __init__(self, x: DataFrame, y: Series, val_split: float, test_split: float,
                 num_workers: int, random_state: int, fold:bool, shuffle: bool, batch_size: int,
                 pin_memory: bool, drop_last: bool):
        from sklearn import preprocessing

        self.dm = DataNormalization()
        self.le = preprocessing.LabelEncoder()

        result = self.split_dataset(x.to_numpy(), self.le.fit_transform(y=y.to_numpy()), val_split, test_split,
                                    random_state, fold)
        dataset = result[0]
        self.size: int = dataset.shape[1]

        super().__init__(
            result[0],
            result[1],
            result[2],
            result[3],
            result[4],
            result[5],
            val_split,
            test_split,
            num_workers,
            random_state,
            shuffle,
            batch_size,
            pin_memory,
            drop_last
        )

        self.predict_dataset = pl_bolts.datamodules.SklearnDataset(self.dm.transform(x.to_numpy(), fold),
                                                                   self.le.transform(y))

        '''
        print(f'train data: \n{self.train_dataset} {range(len(self.train_dataset))}\n'
              f'{[self.train_dataset[i] for i in range(len(self.train_dataset))]}\n'
              f'{self.train_dataset[0]}')
        sys.exit(1)
        '''

    def get_phenotype(self, item):
        return self.le.inverse_transform(item)

    def split_dataset(self, x, y, val_split: float, test_split: float, random_state: int,
                      fold: bool) -> \
            Tuple[Any, Any, Any, Any, Any, Any]:
        holding_split: float = val_split + test_split
        x_train, x_holding, y_train, y_holding = train_test_split(x, y, test_size=holding_split,
                                                                  random_state=random_state, stratify=y)
        self.dm.fit(x_train, fold)
        if holding_split == val_split:
            return (
                self.dm.transform(x_train, fold), y_train,
                self.dm.transform(x_holding, fold), y_holding,
                None, None
            )
        elif holding_split == test_split:
            return (
                self.dm.transform(x_train), y_train,
                None, None,
                self.dm.transform(x_holding), y_holding
            )
        else:
            size: float = val_split / holding_split
            x_val, x_test, y_val, y_test = train_test_split(x_holding, y_holding,
                                                            train_size=size,
                                                            shuffle=self.shuffle,
                                                            random_state=random_state)
            return (
                self.dm.transform(x_train), y_train,
                self.dm.transform(x_val), y_val,
                self.dm.transform(x_test), y_test
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        print(f'train: batch size: {self.batch_size} shuffle: {self.shuffle} '
              f'num_workers: {self.num_workers} drop_last: {self.drop_last} pin_memory: {self.pin_memory}')
        loader = DataLoader(
            self.train_dataset,
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
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
        return loader

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

        return loader

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
                 filter_str: str, num_workers: int, random_state: int, fold: bool,
                 shuffle: bool, drop_last: bool, pin_memory: bool):
        super().__init__()  # I guess this inherits __init__ from super class
        # self.testing_dataset = None
        # self.train_dataset = None
        # self.test_input_list = None
        # self.input_list = None

        self.cyclical = cyclical_lr

        # get normalized data quality control
        x, y = get_data(geno=pd.read_csv(data, index_col=0), filter_str=filter_str, path_to_save_qc=transformed_data)
        self.dataset = GPDataModule(
            x, y, val_split, test_split, num_workers, random_state, fold, shuffle, batch_size, pin_memory, drop_last
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
            nn.Linear(self.input_features, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 512),
            nn.ReLU(True),
        )
        '''
        self.encoder = nn.Sequential(
            nn.Linear(self.input_features, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
        )
        self.encoder = nn.Sequential(
            nn.Linear(self.input_features, self.hidden_layer),
            nn.ReLU(True),
            nn.Linear(self.hidden_layer, self.smallest_layer),
            nn.ReLU(True),
        )
        '''
        # def the decoder function
        self.decoder = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Linear(4096, self.output_features),
            nn.Sigmoid()
        )
        '''
        self.decoder = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 4096),
            nn.ReLU(True),
            nn.Linear(4096, self.output_features),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.smallest_layer, self.hidden_layer),
            nn.ReLU(True),
            nn.Linear(self.hidden_layer, self.output_features),
            nn.Sigmoid()
        )
        '''

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
        # return {'model': coder.detach(), 'loss': loss, "input": x, "'output": output.detach()}
        return {'model': coder.detach(), 'loss': loss}

    # end of training epoch
    def training_epoch_end(self, training_step_outputs):
        # scheduler: CyclicLR = self.lr_schedulers()
        epoch = self.current_epoch
        lr_scheduler: Any
        try:
            lr_scheduler = self.lr_schedulers()[0]
        except TypeError:
            lr_scheduler = self.lr_schedulers()

        # extracting training batch data
        losses: Tensor = get_dict_values_1d('loss', training_step_outputs)
        coder: Tensor = get_dict_values_2d('model', training_step_outputs)
        # target: Tensor = get_dict_values_2d('input', training_step_outputs)
        # output: Tensor = get_dict_values_2d('output', training_step_outputs)

        # ===========save model============
        from datetime import datetime
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
        coder_file = self.save_dir.joinpath(f"{self.model_name}-{epoch}={dt_string}.pt")
        torch.save(coder, coder_file)
        # print(f'\n*** save_dir: {self.save_dir} coder_file: {coder_file} ***\n')
        # save_tensor(x=coder, file=coder_file)
        # np.savetxt(fname=coder_file, X=coder_np, fmt='%f', delimiter=',')

        # ======goodness of fit======
        r2_node: Tensor = self.training_r2score_node.compute()
        print(f"epoch[{epoch + 1:4d}]  "
              f"learning_rate: {lr_scheduler.get_last_lr()[0]:.6f} "
              f"loss: {losses.sum():.6f}  "
              f"r2_mode: {r2_node}",
              end=' ', file=sys.stderr)
        '''
        print(f"epoch[{epoch + 1:4d}]  "
              f"learning_rate: {self.learning_rate:.6f} "
              f"loss: {losses.sum():.6f}  ",
              end=' ', file=sys.stderr)
        '''

        # logging metrics into log file
        self.log('learning_rate', lr_scheduler.get_last_lr()[0])
        self.log('loss', torch.sum(losses))
        self.log('r2score', r2_node)

        # clean up
        del losses
        del r2_node
        del coder
        del coder_file
        self.training_r2score_node.reset()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # define validation step
    def validation_step(self, batch, batch_idx) -> Dict[str, Tensor]:
        x = batch[0]
        output, _ = self.forward(x)

        self.testing_r2score_node.update(preds=output, target=x)
        loss = f.mse_loss(input=output, target=x)
        '''
        print(f'{batch_idx} val step batch size: {self.hparams.batch_size} output dim: {output.size()} '
              f'batch dim: {x.size()} loss dim: {loss.size()}')
        '''
        # return {'loss': loss, 'r2_node': r2_node, 'input': x, 'output': output}
        # return {'loss': loss, "input": x, "output": output}
        return {"loss": loss}

    # end of validation epoch
    def validation_epoch_end(self, testing_step_outputs):
        # extracting training batch data
        loss = get_dict_values_1d('loss', testing_step_outputs)
        # target = get_dict_values_2d('input', testing_step_outputs)
        # output = get_dict_values_2d('output', testing_step_outputs)
        # print(f'regular losses: {losses.size()} pred: {pred.size()} target: {target.size()}')

        # ======goodness of fit ======
        r2_node: Tensor = self.testing_r2score_node.compute()
        '''
        r2_node: Tensor = tm.functional.regression.r2_score(preds=output, target=target,
                                                            multioutput='variance_weighted')
        '''
        print(f"test_loss: {torch.sum(loss):.6f} "
              f"test_r2_node: {r2_node}",
              file=sys.stderr)
        '''
        print(f"test_loss: {torch.sum(loss):.6f} ",
              file=sys.stderr)
        '''
        # logging validation metrics into log file
        self.log('testing_loss', torch.sum(loss), on_step=False, on_epoch=True)
        self.log('testing_r2score', r2_node, on_step=False, on_epoch=True)

        # clean up
        del r2_node
        del loss

        self.testing_r2score_node.reset()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

            # configures the optimizers through learning rate

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler: Any
        if self.cyclical:
            it_per_epoch = math.ceil(len(self.train_dataloader()) / self.hparams.batch_size)
            print(f'it_per_epoch: {it_per_epoch}')
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=self.min_lr,
                                                          mode='exp_range',
                                                          cycle_momentum=False,
                                                          step_size_up=4 * it_per_epoch,
                                                          max_lr=self.learning_rate)
        else:
            scheduler = SWALR(optimizer, swa_lr=self.learning_rate, anneal_epochs=10, anneal_strategy="cos")

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

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
        parser.add_argument("--fold", type=bool, default=False,
                            help='selecting this flag causes the data to be transformed to change fold relative to '
                                 'row median. default is False')
        parser.add_argument("-bs", "--batch_size", type=int, default=64, help='the size of each batch e.g. 64')
        parser.add_argument("-vs", "--val_split", type=float, default=0.1,
                            help='validation set split ratio. default is 0.1')
        parser.add_argument("-ts", "--test_split", type=float, default=0.0,
                            help='test set split ratio. default is 0.0')
        parser.add_argument("-w", "--num_workers", type=int, default=0,
                            help='number of processors used to load data. ie worker = 4 * # of GPU. default is 0')
        parser.add_argument("-f", "--filter_str", nargs="*",
                            help='filter string(s) to select which rows are processed. default: \'\'')
        parser.add_argument("-rs", "--random_state", type=int, default=42,
                            help='sets a seed to the random generator, so that your train-val-test splits are '
                                 'always deterministic. default is 42')
        parser.add_argument("-s", "--shuffle", action='store_true', default=False,
                            help='when this flag is used the dataset is shuffled before splitting the dataset.')
        parser.add_argument("-clr", "--cyclical_lr", action="store_true", default=False,
                            help='when this flag is used cyclical learning rate will be use other stochastic weight '
                                 'average is implored for training.')
        parser.add_argument("--drop_last", action='store_true', default=False,
                            help='selecting this flag causes the last column in the dataset to be dropped.')
        parser.add_argument("--pin_memory", type=bool, default=True,
                            help='selecting this flag causes the numpy to tensor conversion to be less efficient.')

        return parent_parser
