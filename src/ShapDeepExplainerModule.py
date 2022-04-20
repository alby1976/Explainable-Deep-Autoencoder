# python system library
from typing import Optional, Tuple, Union, Sequence, Any

# 3rd party modules
import numpy as np
import pytorch_lightning as pl
import shap
import torch
from numpy import ndarray
from pandas import Series
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch import nn, Tensor

# custom modules
from src.AutoEncoderModule import AutoGenoShallow


class ShapDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        super().__init__()
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        x = self.x[idx].astype(np.float32)
        y = self.y[idx]

        # Do not convert integer to float for classification data
        if not ((y.dtype == np.int32) or (y.dtype == np.int64)):
            y = y.astype(np.float32)

        return x, y


class ShapDataModule(pl.LightningDataModule):
    # this class assumes the data has been cleaned and transformed
    def __init__(self, x: ndarray, hidden: ndarray, labels: Series, test_split: float = 0.2,
                 num_workers: int = 8, random_state: int = 42, shuffle: bool = False, batch_size: int = 64,
                 pin_memory: bool = True, drop_last: bool = False):
        super().__init__()
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.train_dataset, self.test_dataset = self.create_train_test_dataset(x, hidden, labels, test_split,
                                                                               random_state)

    @staticmethod
    def create_train_test_dataset(x, hidden: ndarray, labels: Series, test_split: float = 0.2,
                                  random_state: int = 42) -> Tuple[Any, Any]:
        dataset = ShapDataset(x, hidden)
        unique, unique_count = np.unique(labels, return_counts=True)
        if unique.size > 1 and np.min(unique_count) > 1:
            train_indices, test_indices = train_test_split(list(range(len(labels))), test_size=test_split,
                                                           random_state=random_state, stratify=labels)
        else:
            train_indices, test_indices = train_test_split(list(range(len(labels))), test_size=test_split,
                                                           random_state=random_state)
        return torch.utils.data.Subset(dataset, train_indices), torch.utils.data.Subset(dataset, test_indices)

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

    def val_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def teardown(self, stage: Optional[str] = None):
        # Used to clean-up when the run is finished
        ...

    def get_train(self) -> Tuple[ndarray, ndarray]:
        result =  np.asarray([[x, y] for x, y in self.train_dataset])
        return result[:, 0], result[:, 1]

    def get_test(self) -> Tuple[ndarray, ndarray]:
        result =  np.asarray([[x, y] for x, y in self.test_dataset])
        return result[:, 0], result[:, 1]

    @staticmethod
    def get_x(dataloader: EVAL_DATALOADERS) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
        return torch.cat([x for x, y in dataloader], dim=0)

    @staticmethod
    def get_y(dataloader: EVAL_DATALOADERS) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
        return torch.cat([y for y in dataloader], dim=0)

    #
    # deep_explainer =
    pass


def process_shap_values(trainer: pl.Trainer, model: AutoGenoShallow, x, hidden, labels):
    model.decoder = nn.Identity()

    explainer = shap.DeepExplainer(model, x_train)

    #
    # deep_explainer =
    pass
