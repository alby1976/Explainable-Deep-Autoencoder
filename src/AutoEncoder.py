# ** Use Python to run Deep Autoencoder (feature selection) **
# ** path - is a string to desired path location. **
import sys
from typing import Tuple

import pandas as pd
import pytorch_lightning as pl
from pathlib import Path

from pytorch_lightning.callbacks import ModelSummary, StochasticWeightAveraging
from torchinfo import summary
import torch
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from AutoEncoderModule import AutoGenoShallow
from CommonTools import create_dir
import os


def main(model_name: str, path_to_data: Path, path_to_save_qc: Path, path_to_save_ae: Path,
         compression_ratio: int, num_epochs: int, batch_size: int):
    if not (path_to_data.is_file()):
        print(f'{path_to_data} is not a file')
        sys.exit(-1)

    # instantiate model
    create_dir(path_to_save_ae)
    model = AutoGenoShallow(save_dir=path_to_save_ae, path_to_data=path_to_data, path_to_save_qc=path_to_save_qc,
                            model_name=model_name, compression_ratio=compression_ratio, batch_size=batch_size)
    # find ideal learning rate
    seed_everything(42)
    stop_loss = EarlyStopping(monitor='testing_loss', mode='min', patience=10, verbose=True,
                              check_on_train_epoch_end=False)
    trainer: Trainer
    log_dir = path_to_save_ae.joinpath('log')
    ckpt_dir = path_to_save_ae.joinpath('ckpt')
    create_dir(log_dir)
    create_dir(ckpt_dir)
    csv_logger = CSVLogger(save_dir=str(log_dir), name=model_name)
    tensor_board_logger = TensorBoardLogger(save_dir=str(log_dir), name=model_name)
    if torch.cuda.is_available():
        swa = StochasticWeightAveraging(device='cuda')
        trainer = pl.Trainer(min_epochs=num_epochs,
                             max_epochs=-1,
                             default_root_dir=str(ckpt_dir),
                             log_every_n_steps=1,
                             logger=[csv_logger, tensor_board_logger],
                             deterministic=True,
                             gpus=1,
                             auto_select_gpus=True,
                             stochastic_weight_avg=False,
                             callbacks=[stop_loss, ModelSummary(max_depth=2), swa],
                             # amp_backend="apex",
                             # amp_level="O2",
                             precision=16,
                             # auto_scale_batch_size='binsearch',
                             enable_progress_bar=True)
    else:
        swa = StochasticWeightAveraging(device='cpu')
        trainer = pl.Trainer(min_epochs=num_epochs,
                             max_epochs=-1,
                             default_root_dir=str(ckpt_dir),
                             log_every_n_steps=1,
                             logger=[csv_logger, tensor_board_logger],
                             deterministic=True,
                             stochastic_weight_avg=False,
                             callbacks=[stop_loss, ModelSummary(max_depth=2), swa],
                             # auto_scale_batch_size='binsearch',
                             enable_progress_bar=True)

    print(f'...Finding ideal batch size....')
    trainer.tuner.scale_batch_size(model=model, init_val=model.hparams.batch_size, mode='binsearch')

    print('...Finding ideal learning rate....')
    model.learning_rate = trainer.tuner.lr_find(model).suggestion()
    model.min_lr = model.learning_rate / 6.0
    print(f'min lr: {model.min_lr} max lr: {model.learning_rate}')

    # train & validate model
    print(f'...Training and Validating model...')
    trainer.fit(model=model)
    print('f...Model Summary')
    tmp: Tuple[int, int] = pd.read_csv(path_to_data, index_col=0).shape
    summary(model, (tmp[0], model.input_features))


if __name__ == '__main__':
    if torch.cuda.is_available():
        # device_index = "0"
        # os.environ["CUDA_VISIBLE_DEVICES"] = device_index
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = 'true'

    if len(sys.argv) < 7:
        print('Default setting are used. Either change AutoEncoder.py to change settings or type:\n')
        print('python AutoEncoder.py model_name original_datafile '
              'quality_control_filename dir_AE_model compression_ratio epoch batch_size')
        print('\tmodel_name - model name e.g. AE_Geno')
        print('\toriginal_datafile - original datafile e.g. ../data_example.csv')
        print('\tquality_control_filename - filename of original data after quality control e.g. ./data_QC.csv')
        print('\tdir_AE_model - base dir to saved AE models e.g. ./AE')
        print('\tcompression_ratio - compression ratio for smallest layer NB: ideally a number that is power of 2')
        print('\tnum_epoch - min number of epochs e.g. 200')
        print('\tbatch_size - the size of each batch e.g. 4096')

        main('AE_Geno', Path('../data_example.csv'), Path('./data_QC.csv'), Path('./AE'), 32, 200, 4096)
    else:
        main(model_name=sys.argv[1], path_to_data=Path(sys.argv[2]), path_to_save_qc=Path(sys.argv[3]),
             path_to_save_ae=Path(sys.argv[4]),
             compression_ratio=int(sys.argv[5]), num_epochs=int(sys.argv[6]), batch_size=int(sys.argv[7]))
