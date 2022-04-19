# python system library
from typing import Optional, Tuple, Union, Sequence

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


# custom modules


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
    def __init__(self, x: ndarray, hidden: ndarray, labels: Series, test_split: float,
                 num_workers: int, random_state: int, shuffle: bool, batch_size: int,
                 pin_memory: bool, drop_last: bool):
        super().__init__()
        dataset = ShapDataset(x, hidden)
        unique, unique_count = np.unique(labels, return_counts=True)

        if unique.size > 1 and np.min(unique_count) > 1:
            train_indices, test_indices = train_test_split(list(range(len(labels))), test_size=test_split,
                                                           random_state=random_state, stratify=labels)
        else:
            train_indices, test_indices = train_test_split(list(range(len(labels))), test_size=test_split,
                                                           random_state=random_state)

        self.train_dataset = torch.utils.data.Subset(dataset, train_indices)
        self.test_dataset = torch.utils.data.Subset(dataset, test_indices)
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.drop_last = drop_last

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

    @staticmethod
    def get_x(dataloader: EVAL_DATALOADERS) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
        return torch.cat([x for x, y in dataloader], dim=0)

    @staticmethod
    def get_y(dataloader: EVAL_DATALOADERS) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
        return torch.cat([y for y in dataloader], dim=0)


class ShapDeepExplainer:
    def __int__(self, model, data, x=None):
        super().__init__()
        self.explainer = shap.DeepExplainer(model, data)
        self.shap_values = self.explainer.shap_values(x) if x is not None else None

    def eval(self, x_test=None):
        if self.shap_values is None and x_test is None:
            raise NameError "No values was given to shap.shap_values to evaluate"


