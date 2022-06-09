# ** Use Python to run Deep Autoencoder (feature selection) **
# ** path - is a string to desired path location. **
import argparse
import gc
import os
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List, Union, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelSummary, StochasticWeightAveraging, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from AutoEncoderModule import AutoGenoShallow
from CommonTools import create_dir, float_or_none
from SHAP_combo import add_shap_arguments, create_shap_tree_val
from ShapDeepExplainerModule import create_shap_values

_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def main(args):
    if not args.data.is_file():
        print(f'{args.data} is not a file')
        sys.exit(-1)

    with wandb.init(name=args.name, project="XAE4Exp", config=args):
        # wandb configuration
        wandb.init()
        # pathway instantiate model
        path_to_save_ae = Path(args.save_dir)
        create_dir(path_to_save_ae)
        model = AutoGenoShallow(args.save_dir, args.name, args.smallest_layer, args.cyclical_lr,
                                args.learning_rate, args.reg_param, args.data,
                                args.transformed_data, args.batch_size, args.val_split, args.test_split,
                                args.filter_str,
                                args.num_workers, args.random_state, args.fold, args.shuffle, args.drop_last,
                                args.pin_memory_no, args.verbose, args.ensembl_version)

        seed_everything(args.random_state)
        trainer: Trainer
        log_dir = path_to_save_ae.joinpath('log')
        ckpt_dir = path_to_save_ae.joinpath('ckpt')
        create_dir(log_dir)
        create_dir(ckpt_dir)

        learning_rate_monitor = LearningRateMonitor(logging_interval='step', log_momentum=True)
        wandb_logger = WandbLogger(name=args.name, log_model=True)

        ckpt: ModelCheckpoint = ModelCheckpoint(dirpath=ckpt_dir,
                                                filename=args.name + '-{epoch}-{testing_loss:.6f}',
                                                monitor=args.monitor, mode=args.mode, verbose=args.verbose,
                                                save_top_k=1)
        stop_loss = EarlyStopping(monitor=args.monitor, mode=args.mode, patience=args.patience, verbose=args.verbose,
                                  check_on_train_epoch_end=args.check_on_train_epoch_end)
        '''
        stop_loss = EarlyStopping(monitor='testing_r2score', mode='max', patience=10, verbose=True,
                                  check_on_train_epoch_end=False)
        '''

        callbacks: List[
            Union[EarlyStopping, ModelSummary, LearningRateMonitor, StochasticWeightAveraging, ModelCheckpoint]]
        if args.cyclical_lr:
            callbacks = [stop_loss, ModelSummary(max_depth=2), learning_rate_monitor, ckpt]
        else:
            swa_module: StochasticWeightAveraging = StochasticWeightAveraging(device=_DEVICE)
            callbacks = [stop_loss, ModelSummary(max_depth=2), learning_rate_monitor, ckpt, swa_module]

        if torch.cuda.is_available():
            trainer = pl.Trainer.from_argparse_args(args, max_epochs=-1,
                                                    default_root_dir=str(ckpt_dir),
                                                    log_every_n_steps=1,
                                                    logger=wandb_logger,
                                                    deterministic=True,
                                                    gpus=1,
                                                    auto_select_gpus=True,
                                                    callbacks=callbacks,
                                                    # amp_backend="apex",
                                                    # amp_level="O2",
                                                    precision=16,
                                                    auto_lr_find=args.tune,
                                                    # auto_scale_batch_size='binsearch',
                                                    enable_progress_bar=False)
        else:
            trainer = pl.Trainer.from_argparse_args(args,
                                                    max_epochs=-1,
                                                    default_root_dir=str(ckpt_dir),
                                                    log_every_n_steps=1,
                                                    logger=wandb_logger,
                                                    deterministic=True,
                                                    callbacks=callbacks,
                                                    auto_lr_find=args.tune,
                                                    # auto_scale_batch_size='binsearch',
                                                    enable_progress_bar=False)

        # find ideal learning rate and batch_size
        if args.tune:
            print(f'...Finding ideal batch size & learning rate....')
            print(f'starting batch size: {model.hparams.batch_size}')
            trainer.tuner.scale_batch_size(model=model, init_val=model.hparams.batch_size, mode='binsearch')
            if model.lr > 0.001:
                model.lr = 0.00008
            print(f'min lr: {model.lr / 6.0} max lr: {model.lr}')

        # train & validate model
        print(f'...Training and Validating model...')
        trainer.fit(model=model)
        hidden_layer = trainer.predict(model=model, ckpt_path="best")
        df = None

        if hidden_layer is not None:
            hidden_layer = torch.cat([hidden_layer[i] for i in range(len(hidden_layer))])
            hidden_layer = hidden_layer.detach().cpu().numpy()
            np.savetxt(fname=args.save_dir.joinpath(f"{args.name}-output.csv"), X=hidden_layer, fmt='%f', delimiter=',')
            df = pd.DataFrame(hidden_layer)
            tbl = wandb.Table(dataframe=df, dtype=float)
            wandb.log({"AE_out": tbl})

        del trainer
        del tbl

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if args.deep:
            create_shap_values(model, args.name + "_Shap", args.save_dir.joinpath(args.gene_model),
                               args.save_dir.joinpath(args.save_bar),
                               args.save_dir.joinpath(args.save_scatter), args.top_rate)
        else:
            mask: List[bool] = model.dataset.dm.column_mask
            create_shap_tree_val(args.name + "_Shap", model.dataset.dm,
                                 model.dataset.y.to_numpy(), model.dataset.x, model.dataset.gene_names[mask], df,
                                 args.save_dir.joinpath(args.save_bar),
                                 args.save_dir.joinpath(args.save_scatter), args.save_dir.joinpath(args.gene_model),
                                 args.num_workers, args.fold, args.val_split,
                                 args.random_state, args.shuffle, args.boost, args.top_rate)

        wandb.finish()


if __name__ == '__main__':
    if torch.cuda.is_available():
        # device_index = "0"
        # os.environ["CUDA_VISIBLE_DEVICES"] = device_index
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = 'true'
    parser: ArgumentParser = argparse.ArgumentParser(description='Generate AE Model that can used Shape to predict the '
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
    parser.add_argument("--ensembl_version", type=int, default=104, help='Ensembl Release version e.g. 104')
    parser.add_argument("-d", "--deep", action="store_true", default=False,
                        help="using this flag selects DeepExplainer otherwise TreeExplainer is used.")

    # add model specific args
    parser = AutoGenoShallow.add_model_specific_args(parser)

    # add all the available trainer options to argparse
    # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    parser = Trainer.add_argparse_args(parser)

    # add SHAP arguments
    group = parser.add_argument_group("calculates the shapey values for the AE model's output")
    add_shap_arguments(group)

    # parse the command line arguments
    arguments: Namespace = parser.parse_args()
    print(arguments)

    main(arguments)
