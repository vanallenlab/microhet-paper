import pandas as pd
import numpy as np
import sys
sys.path.append('~/pytorch-lightning')
from argparse import ArgumentParser

import pytorch_lightning as pl

from mc_lightning.utilities import make_folds
from datamodules import SlideDataModule
from generic_transforms import RGBEvalTransform, RGBTrainTransform
from base_models import PretrainedResnet50FT

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

    # add additional args to look for without clogging above
    parser = pl.Trainer.add_argparse_args(parser)
    parser = PretrainedResnet50FT.add_model_specific_args(parser)
    parser = SlideDataModule.add_argparse_args(parser)

    args = parser.parse_args()

    # https://pytorch-lightning.readthedocs.io/en/latest/trainer.html?highlight=seed_everything#reproducibility
    # use seed_everything and Trainer(deterministic=True) to fix across numpy, torch, python.random and PYTHONHASHSEED.
    pl.seed_everything(args.seed)

    try:
        paths_df = pd.read_pickle(args.paths_df)
    except:
        paths_df = pd.read_csv(args.paths_df)

    print('should only see this ONCE')

    # TODO does pl.seed_everything reach into calls made like `make_folds`? Instinct says no?
    id_agg = make_folds(args.paths_df, args.out_dir, args.label_var, test_size=0.15, folds=args.folds,
                        seed=args.seed, num_classes=args.num_classes)


    print(f'====== Fold {args.fold_index} ======')
    # create datamodule
    dm = SlideDataModule.from_argparse_args(
        args,
        data_df=paths_df,
        train_ids=id_agg['train_ids'][args.fold_index],
        val_ids=id_agg['val_ids'][args.fold_index],
        test_ids=id_agg['test_ids'],
        train_transform=RGBTrainTransform(args.tile_size, args.crop_size, args.aug_strength),
        eval_transform=RGBEvalTransform(args.tile_size, args.crop_size),
    )

    # create model
    model = PretrainedResnet50FT(args)


    fold_out_dir = f'./{args.fold_index}'
    try:
        os.mkdir(fold_out_dir)
    except:
        pass
    # create trainer
    # TODO add callbacks
    trainer = pl.Trainer.from_argparse_args(args, deterministic=True)

    # fit model
    trainer.fit(model, dm)

    # (optional) test best checkpoint
    trainer.test(datamodule=dm)


if __name__ == '__main__':
    cli_main()
