# ** Use Python to run Deep Autoencoder (feature selection) **
# ** path - is a string to desired path location. **
import argparse
import os
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelSummary, StochasticWeightAveraging, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from AutoEncoderModule import AutoGenoShallow
from CommonTools import create_dir

_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    parser: ArgumentParser = argparse.ArgumentParser(description='Generate AE Model that can used Shape to predit the '
                                                                 'hidden layer features.')

    # add PROGRAM level args
    parser.add_argument('--tune', action='store_true', default=False,
                        help='including this flag causes pytorch lightning to find optimum '
                             'batch_size and learning rate')

    # add model specific args
    parser = AutoGenoShallow.add_model_specific_args(parser)

    # add all the available trainer options to argparse
    # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    print(args)
    if not args.data.is_file():
        print(f'{args.data} is not a file')
        sys.exit(-1)

    # instantiate model
    path_to_save_ae = Path(args.save_dir)
    create_dir(path_to_save_ae)
    model = AutoGenoShallow(args.save_dir, args.name, args.ratio, args.cyclical_lr, args.learning_rate, args.data,
                            args.transformed_data, args.batch_size, args.val_split, args.test_split, args.filter_str,
                            args.num_workers, args.random_state, args.shuffle, args.drop_last, args.pin_memory)

    seed_everything(args.random_state)
    stop_loss = EarlyStopping(monitor='testing_loss', mode='min', patience=10, verbose=True,
                              check_on_train_epoch_end=False)
    trainer: Trainer
    log_dir = path_to_save_ae.joinpath('log')
    ckpt_dir = path_to_save_ae.joinpath('ckpt')
    create_dir(log_dir)
    create_dir(ckpt_dir)
    csv_logger = CSVLogger(save_dir=str(log_dir), name=args.name)
    learning_rate_monitor = LearningRateMonitor(logging_interval='epoch')
    tensor_board_logger = TensorBoardLogger(save_dir=str(log_dir), name=args.name)
    callbacks: List[Union[EarlyStopping, ModelSummary, LearningRateMonitor, StochasticWeightAveraging]]
    swa: bool = False
    if args.cyclical_lr:
        callbacks = [stop_loss, ModelSummary(max_depth=2), learning_rate_monitor]
    else:
        swa = True
        swa_module: StochasticWeightAveraging = StochasticWeightAveraging(device=_DEVICE)
        callbacks = [stop_loss, ModelSummary(max_depth=2), learning_rate_monitor, swa_module]

    if torch.cuda.is_available():
        trainer = pl.Trainer.from_argparse_args(args, max_epochs=-1,
                                                default_root_dir=str(ckpt_dir),
                                                log_every_n_steps=1,
                                                logger=[csv_logger, tensor_board_logger],
                                                deterministic=True,
                                                gpus=1,
                                                auto_select_gpus=True,
                                                stochastic_weight_avg=swa,
                                                callbacks=callbacks,
                                                # amp_backend="apex",
                                                # amp_level="O2",
                                                precision=16,
                                                # auto_scale_batch_size='binsearch',
                                                enable_progress_bar=False)
    else:
        trainer = pl.Trainer.from_argparse_args(args,
                                                max_epochs=-1,
                                                default_root_dir=str(ckpt_dir),
                                                log_every_n_steps=1,
                                                logger=[csv_logger, tensor_board_logger],
                                                deterministic=True,
                                                stochastic_weight_avg=swa,
                                                callbacks=callbacks,
                                                # auto_scale_batch_size='binsearch',
                                                enable_progress_bar=False)

    # find ideal learning rate and batch_size
    if args.tune:
        print(f'...Finding ideal batch size....')
        print(f'starting batch size: {model.hparams.batch_size}')
        trainer.tuner.scale_batch_size(model=model, init_val=model.hparams.batch_size, mode='binsearch')
        torch.cuda.empty_cache()

        print('...Finding ideal learning rate....')
        model.learning_rate = trainer.tuner.lr_find(model).suggestion()
        model.min_lr = model.learning_rate / 6.0
        print(f'min lr: {model.min_lr} max lr: {model.learning_rate}')
        torch.cuda.empty_cache()

    # train & validate model
    print(f'...Training and Validating model...')
    trainer.fit(model=model)


if __name__ == '__main__':
    if torch.cuda.is_available():
        # device_index = "0"
        # os.environ["CUDA_VISIBLE_DEVICES"] = device_index
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = 'true'

    main()

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
