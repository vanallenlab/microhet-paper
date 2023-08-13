import pandas as pd
import numpy as np
import sys
sys.path.append('~/pytorch-lightning')
from argparse import ArgumentParser
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from torch.utils.data import random_split, DataLoader

from mc_lightning.utilities.utilities import tile_sampler, subsample_tiles
from mc_lightning.models.resnet.resnet_module import PretrainedResnet50FT
from mc_lightning.models.resnet.resnet_transforms import RGBTrainTransform, RGBEvalTransform
from mc_lightning.models.resnet.resnet_dataset import SlideDataset

def cli_main():
    parser = ArgumentParser()
    # TODO add an input format flag to determine whether to use a `.pth` or `.jpeg` input in `SlideDataModule` loading
    parser.add_argument('--out_dir', '-o', type=str)
    parser.add_argument('--seed', type=int, default=None, metavar='N',
                        help='set a random seed for torch and numpy (default: None)')
    parser.add_argument('-df', '--paths_df', type=str,
                        help='file path of dataframe with ids/tile paths [pickle or csv]')
    parser.add_argument('--prefix', type=str, default='20x_512px_',
                        help='prefix to append to output ID csv')
    parser.add_argument('--folds', type=int, default=4)
    parser.add_argument('--fold_index', type=int, default=0)

    parser.add_argument('--train_ids', type=str, help='file path of list of IDs in train set')
    parser.add_argument('--dev_ids', type=str, help='file path of list of IDs in validation (dev) set')

    parser.add_argument('--label_var', type=str)
    parser.add_argument('--slide_var', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('-tps', '--tiles_per_slide', type=int, default=50,
                        help='if specified, num. tiles to sample per slide')
    parser.add_argument('--slide_var_name', type=str, default='slide_id')
    parser.add_argument('--tile_size', type=int, help='Uncropped input tile size', default=512)
    parser.add_argument('--crop_size', type=int, help='Uncropped input tile size', default=224)
    parser.add_argument('--aug_strength', '-s', type=float, default=1.0)
    parser.add_argument('--run_test_set', '-test', action='store_true')


    # add additional args to look for without clogging above
    parser = pl.Trainer.add_argparse_args(parser)
    parser = PretrainedResnet50FT.add_model_specific_args(parser)

    args = parser.parse_args()

    # https://pytorch-lightning.readthedocs.io/en/latest/trainer.html?highlight=seed_everything#reproducibility
    # use seed_everything and Trainer(deterministic=True) to fix across numpy, torch, python.random and PYTHONHASHSEED.
    # pl.seed_everything(args.seed)

    try:
        paths_df = pd.read_pickle(args.paths_df)
    except:
        paths_df = pd.read_csv(args.paths_df)

    fold_idx = args.fold_index
    print(f'====== Only Running on Fold {fold_idx} ======')
    # PULLING FROM model-comparison train_folds.py
    temp_path = os.path.join(args.out_dir, args.prefix + f'fold{fold_idx}_train_slide_ids.csv')
    train_ids = pd.read_csv(temp_path).iloc[:, 1].values

    temp_path = os.path.join(args.out_dir, args.prefix + f'fold{fold_idx}_val_slide_ids.csv')
    val_ids = pd.read_csv(temp_path).iloc[:, 1].values

    try:
        temp_path = os.path.join(args.out_dir, args.prefix + f'test_slide_ids.csv')
        test_ids = pd.read_csv(temp_path).iloc[:, 1].values
    except:
        print('Could not load test set IDs -- if this is expected, ignore')

    if args.tiles_per_slide != -1:
        train_paths = subsample_tiles(paths_df, train_ids, args.tiles_per_slide, args.label_var)
        val_paths = subsample_tiles(paths_df, val_ids, args.tiles_per_slide, args.label_var)
        # test_paths = subsample_tiles(paths_df, test_ids, args.tiles_per_slide, args.label_var)

    train_dataset = SlideDataset(
        paths=train_paths.full_path.values,
        slide_ids=train_paths.index.values,
        labels=train_paths[args.label_var].values,
        transform_compose=RGBTrainTransform(args.tile_size, args.crop_size, args.aug_strength)
    )
    val_dataset = SlideDataset(
        paths=val_paths.full_path.values,
        slide_ids=val_paths.index.values,
        labels=val_paths[args.label_var].values,
        transform_compose=RGBEvalTransform(args.tile_size, args.crop_size)
    )

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  pin_memory=True, num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
                                pin_memory=True, num_workers=args.num_workers)

    # create model
    model = PretrainedResnet50FT(args)
                                                       
    # create trainer
    #trainer = pl.Trainer.from_argparse_args(args, logger=tb_logger, checkpoint_callback=checkpoint_callback)
    trainer = pl.Trainer.from_argparse_args(args)

    # fit model
    trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)


    # If flagged, create test subset and evaluate
    if args.run_test_set:
        test_paths = subsample_tiles(paths_df, test_ids, args.tiles_per_slide, args.label_var)
        test_dataset = SlideDataset(
            paths=test_paths.full_path.values,
            slide_ids=test_paths.index.values,
            labels=test_paths[args.label_var].values,
            transform_compose=RGBEvalTransform(args.tile_size, args.crop_size)
        )
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, pin_memory=True,
                                     num_workers=args.num_workers)
        trainer.test(model, test_dataloaders=test_dataloader)


if __name__ == '__main__':
    cli_main()
