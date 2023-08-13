import pandas as pd
import numpy as np
import sys
sys.path.append('~/pytorch-lightning')
from argparse import ArgumentParser

import pytorch_lightning as pl

from mc_lightning.utilities import make_folds
#from datamodules import SlideDataModule
from datamodules_tensorinput import SlideDataModule
from generic_transforms import RGBEvalTransform, RGBTrainTransform
from base_models import PretrainedResnet50FT

def cli_main():
    parser = ArgumentParser()
    parser.add_argument('--out_dir', '-o', type=str)
    parser.add_argument('--seed', type=int, default=None, metavar='N',
                        help='set a random seed for torch and numpy (default: None)')
    parser.add_argument('-df', '--paths_df', type=str,
                        help='file path of dataframe with ids/tile paths [pickle or csv]')
    parser.add_argument('--prefix', type=str, default='20x_512px_',
                        help='prefix to append to output ID csv')
    parser.add_argument('--folds', type=int)
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

    # TODO does pl.seed_everything reach into calls made like `make_folds`? Instinct says no?
    id_agg = make_folds(args.paths_df, args.out_dir, args.label_var, test_size=0.15, folds=args.folds,
                        seed=args.seed, num_classes=args.num_classes)


    for fold_idx in range(args.folds):
        print(f'====== {fold_idx} ======')
        # create datamodule
        dm = SlideDataModule.from_argparse_args(
            args,
            data_df=paths_df,
            train_ids=id_agg['train_ids'][fold_idx],
            val_ids=id_agg['val_ids'][fold_idx],
            test_ids=id_agg['test_ids'],
            train_transform=RGBTrainTransform(args.tile_size, args.crop_size, args.aug_strength),
            eval_transform=RGBEvalTransform(args.tile_size, args.crop_size),
        )

        # create model
        model = PretrainedResnet50FT(args)


        fold_out_dir = f'./{fold_idx}'
        try:
            os.mkdir(fold_out_dir)
        except:
            pass
        # create trainer
        # TODO add callbacks
        trainer = pl.Trainer.from_argparse_args(args, deterministic=True)
        #trainer = pl.Trainer.from_argparse_args(args, deterministic=True, default_root_dir=fold_out_dir)

        # fit model
        trainer.fit(model, dm)

        # (optional) test best checkpoint
        trainer.test(datamodule=dm)


    # # init default datamodule
    # if args.dataset == 'cifar10':
    #     dm = CIFAR10DataModule.from_argparse_args(args)
    #     dm.train_transforms = SimCLRTrainDataTransform(32)
    #     dm.val_transforms = SimCLREvalDataTransform(32)
    #     args.num_samples = dm.num_samples
    #
    # elif args.dataset == 'stl10':
    #     dm = STL10DataModule.from_argparse_args(args)
    #     dm.train_dataloader = dm.train_dataloader_mixed
    #     dm.val_dataloader = dm.val_dataloader_mixed
    #     args.num_samples = dm.num_unlabeled_samples
    #
    #     (c, h, w) = dm.size()
    #     dm.train_transforms = SimCLRTrainDataTransform(h)
    #     dm.val_transforms = SimCLREvalDataTransform(h)
    #
    # elif args.dataset == 'imagenet2012':
    #     dm = ImagenetDataModule.from_argparse_args(args, image_size=196)
    #     (c, h, w) = dm.size()
    #     dm.train_transforms = SimCLRTrainDataTransform(h)
    #     dm.val_transforms = SimCLREvalDataTransform(h)

    # model = SimCLR(**args.__dict__)
    #
    # # finetune in real-time
    # def to_device(batch, device):
    #     (x1, x2), y = batch
    #     x1 = x1.to(device)
    #     y = y.to(device)
    #     return x1, y
    #
    # online_eval = SSLOnlineEvaluator(z_dim=2048 * 2 * 2, num_classes=dm.num_classes)
    # online_eval.to_device = to_device
    #
    # trainer = pl.Trainer.from_argparse_args(args, callbacks=[online_eval])
    # trainer.fit(model, dm, deterministic=True)


if __name__ == '__main__':
    cli_main()
