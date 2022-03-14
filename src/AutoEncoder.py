# ** Use Python to run Deep Autoencoder (feature selection) **
# ** path - is a string to desired path location. **
import argparse
import os
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelSummary, StochasticWeightAveraging
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from torchinfo import summary

from AutoEncoderModule import AutoGenoShallow, GPDataSet
from CommonTools import create_dir


def main(args: ArgumentParser):
    if not (Path(args.data).is_file()):
        print(f'{args.data} is not a file')
        sys.exit(-1)

    # instantiate model
    path_to_save_ae = Path(args.save_dir)
    create_dir(path_to_save_ae)
    model = AutoGenoShallow(args)
    # find ideal learning rate
    seed_everything(42)
    stop_loss = EarlyStopping(monitor='testing_loss', mode='min', patience=10, verbose=True,
                              check_on_train_epoch_end=False)
    trainer: Trainer
    log_dir = path_to_save_ae.joinpath('log')
    ckpt_dir = path_to_save_ae.joinpath('ckpt')
    create_dir(log_dir)
    create_dir(ckpt_dir)
    csv_logger = CSVLogger(save_dir=str(log_dir), name=args.name)
    tensor_board_logger = TensorBoardLogger(save_dir=str(log_dir), name=args.name)
    if torch.cuda.is_available():
        swa = StochasticWeightAveraging(device='cuda')
        trainer = pl.Trainer(max_epochs=-1,
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
        trainer = pl.Trainer(args,
                             max_epochs=-1,
                             default_root_dir=str(ckpt_dir),
                             log_every_n_steps=1,
                             logger=[csv_logger, tensor_board_logger],
                             deterministic=True,
                             stochastic_weight_avg=False,
                             callbacks=[stop_loss, ModelSummary(max_depth=2), swa],
                             # auto_scale_batch_size='binsearch',
                             enable_progress_bar=True)

    if args.tune:
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

    parser: ArgumentParser = argparse.ArgumentParser(description='Generate AE Model that can used Shape to predit the '
                                                                 'hidden layer features.')

    # add PROGRAM level args
    parser.add_argument('--tune', type=bool, default=True, help='whether or not pytorch lightning find optimum '
                                                                'batch_size and learning rate')

    # add model specific args
    parser = AutoGenoShallow.add_model_specific_args(parser)

    # add DataModule specific args
    parser = GPDataSet.add_model_specific_args(parser)

    # add all the available trainer options to argparse
    # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    parser = Trainer.add_argparse_args(parser)
    print(parser.parse_args())
    sys.exit(-1)
    main(parser.parse_args())

    '''
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
    '''
