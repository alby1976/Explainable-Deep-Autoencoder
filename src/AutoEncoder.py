# ** Use Python to run Deep Autoencoder (feature selection) **
# ** path - is a string to desired path location. **
import argparse
import os
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List, Union

import pandas as pd
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelSummary, StochasticWeightAveraging, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from AutoEncoderModule import AutoGenoShallow
from CommonTools import create_dir, float_or_none

_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def main(args):
    if not args.data.is_file():
        print(f'{args.data} is not a file')
        sys.exit(-1)

    with wandb.init(config=args):
        # instantiate model
        path_to_save_ae = Path(args.save_dir)
        create_dir(path_to_save_ae)
        model = AutoGenoShallow(args.save_dir, args.name, args.ratio, args.cyclical_lr, args.learning_rate, args.data,
                                args.transformed_data, args.batch_size, args.val_split, args.test_split, args.filter_str,
                                args.num_workers, args.random_state, args.fold, args.shuffle, args.drop_last, args.pin_memory)

        seed_everything(args.random_state)
        trainer: Trainer
        log_dir = path_to_save_ae.joinpath('log')
        wandb_dir = path_to_save_ae.joinpath('log/wandb')
        ckpt_dir = path_to_save_ae.joinpath('ckpt')
        create_dir(log_dir)
        create_dir(wandb_dir)
        create_dir(ckpt_dir)
        csv_logger = CSVLogger(save_dir=str(log_dir), name=args.name)

        learning_rate_monitor = LearningRateMonitor(logging_interval='epoch')
        wandb_logger = WandbLogger(name=args.name, save_dir=str(wandb_dir), log_model=True)
        # wandb.config.update(args)  # adds all of the arguments as config variables

        ckpt: ModelCheckpoint = ModelCheckpoint(dirpath=ckpt_dir,
                                                filename='best-{epoch}-{testing_loss:.6f}',
                                                monitor=args.monitor, mode=args.mode, verbose=args.verbose, save_top_k=1)
        stop_loss = EarlyStopping(monitor=args.monitor, mode=args.mode, patience=args.patience, verbose=args.verbose,
                                  check_on_train_epoch_end=args.check_on_train_epoch_end)
        '''
        stop_loss = EarlyStopping(monitor='testing_r2score', mode='max', patience=10, verbose=True,
                                  check_on_train_epoch_end=False)
        '''
        callbacks: List[Union[EarlyStopping, ModelSummary, LearningRateMonitor, StochasticWeightAveraging, ModelCheckpoint]]
        swa: bool = False
        if args.cyclical_lr:
            callbacks = [stop_loss, ModelSummary(max_depth=2), learning_rate_monitor, ckpt]
        else:
            swa = True
            swa_module: StochasticWeightAveraging = StochasticWeightAveraging(device=_DEVICE)
            callbacks = [stop_loss, ModelSummary(max_depth=2), learning_rate_monitor, ckpt, swa_module]

        if torch.cuda.is_available():
            trainer = pl.Trainer.from_argparse_args(args, max_epochs=-1,
                                                    default_root_dir=str(ckpt_dir),
                                                    log_every_n_steps=1,
                                                    logger=[csv_logger, wandb_logger],
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
                                                    logger=[csv_logger, wandb_logger],
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
        output = trainer.predict(model=model)
        if output is not None:
            df = pd.DataFrame(output)
            df.to.csv(args.save_dir.joinpath(f"{args.model_name}-final"))


if __name__ == '__main__':
    if torch.cuda.is_available():
        # device_index = "0"
        # os.environ["CUDA_VISIBLE_DEVICES"] = device_index
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = 'true'
    parser: ArgumentParser = argparse.ArgumentParser(description='Generate AE Model that can used Shape to predit the '
                                                                 'hidden layer features.')

    # add PROGRAM level args
    parser.add_argument('--tune', action='store_true', default=False,
                        help='including this flag causes pytorch lightning to find optimum '
                             'batch_size and learning rate')
    # add EarlyStop parameters
    # stop_loss = EarlyStopping(monitor='testing_loss', mode='min', patience=10, verbose=True,
    #                          check_on_train_epoch_end=False)
    """
    EarlyStopping(monitor=None, min_delta=0.0, patience=3, verbose=False, mode='min', strict=True, check_finite=True,
                  stopping_threshold=None, divergence_threshold=None, check_on_train_epoch_end=None)
    """
    parser.add_argument("--min_delta", type=float, default=0,
                        help="minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute "
                             "change of less than or equal to min_delta, will count as no improvement.")
    parser.add_argument("--stopping_threshold", type=float_or_none, default='None',
                        help="Stop training immediately once the monitored quantity reaches this threshold.")
    parser.add_argument("--divergence_threshold", type=float_or_none, default='None',
                        help="Stop training as soon as the monitored quantity becomes worse than this threshold.")
    parser.add_argument("--check_on_train_epoch_end", action='store_true', default=False,
                        help="whether to run early stopping at the end of the training epoch. If this is False, then "
                             "the check runs at the end of the validation.")
    parser.add_argument("--monitor", type=str, default="testing_loss",
                        help="the metric to monitor. eg. 'testing_loss'")
    parser.add_argument("--mode",
                        default='min',
                        const='min',
                        nargs='?',
                        choices=['min', 'max'],
                        help="In 'min' mode, training will stop when the quantity monitored has stopped decreasing and "
                             "in 'max' mode it will stop when the quantity monitored has stopped increasing. "
                             "(default: %(default)s)")
    parser.add_argument("-p", "--patience", type=int, default=10,
                        help="the number of metric checks before this module assume no change and trigger early stop"
                             "\n\n"
                             "It must be noted that the patience parameter counts the number of validation checks "
                             "with no improvement and would not be the number of training epoch if "
                             "pytorch_lightning.trainer.Trainer.params.check_val_every_n_epoch is not 1."
                             "i.e. pytorch_lightning.trainer.Trainer.params.check_val_every_n_epoch=2 and patience is 3"
                             "then at least 6 training of no improvement before training will stop.")
    parser.add_argument("-v", "--verbose", action="store_true", default=False, help="verbosity mode")

    # add model specific args
    parser = AutoGenoShallow.add_model_specific_args(parser)

    # add all the available trainer options to argparse
    # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    parser = Trainer.add_argparse_args(parser)

    # parse the command line arguements
    arguments: Namespace = parser.parse_args()
    print(arguments)

    main(arguments)

    '''
    if len(sys.argv) < 7:
        print('Default setting are used. Either change AutoEncoder.py to change settings or type:\n')
        print('python AutoEncoder.py model_name original_datafile '
              'quality_control_filename dir_AE_model compression_ratio epoch batch_size')
        print('\tmodel_name - model name e.g. AE_Geno')
        print('\toriginal_datafile - original datafile e.g. ../data_example.csv')
        print('\tquality_control_filename - filename of original x after quality control e.g. ./data_QC.csv')
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
