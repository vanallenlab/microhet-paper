import torch
import torch.utils.data as data
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from torchvision import transforms

from PIL import Image
from mc_lightning.utilities import pil_loader

class SlideDataset(data.Dataset):
    """
    Modification of vanilla `tmb_bot.utilities.Dataset` class to facilitate having
    a label for classification as well as the slide name itself
    """
    def __init__(self, paths, slide_ids, labels, transform_compose):
        """
        Paths and labels should be array like
        """
        self.paths = paths
        self.slide_ids = slide_ids
        self.labels = labels
        self.transform = transform_compose
        self.to_pil = transforms.ToPILImage()

    def __len__(self):
        return self.paths.shape[0]

    def __getitem__(self, index):
        'Generates one sample of data'
        img_path = self.paths[index]
        tensor_file = torch.from_numpy(torch.load(img_path)).permute(2,0,1)
        pil_file = self.to_pil(tensor_file)
        pil_file = self.transform(pil_file)
        slide_id = self.slide_ids[index]
        label = self.labels[index]

        return pil_file, label, slide_id


class SlideDataModule(pl.LightningDataModule):
#    def __init__(self, data_df, train_ids, val_ids, test_ids, train_transform, eval_transform,
 #                label_var='angio_high', slide_var='slide_id',
  #               tile_size=512, workers=8, batch_size=100, tiles_per_slide=500):
    def __init__(self, data_df, train_ids, val_ids, test_ids, train_transform, eval_transform,
                 label_var, slide_var, tile_size, num_workers, batch_size, tiles_per_slide):
        super().__init__()
        self.data_df = data_df
        self.train_ids = train_ids
        self.val_ids = val_ids
        self.test_ids = test_ids

        self.train_transform = train_transform
        self.eval_transform = eval_transform

        self.label_var = label_var
        self.slide_var = slide_var

        self.workers = num_workers
        self.batch_size = batch_size
        self.tiles_per_slide = tiles_per_slide

        self.dims = (3, tile_size, tile_size)

    def tile_sampler(self, x):
        samples = x.sample(min(len(x), self.tiles_per_slide))
        return samples

    def subsample_tiles(self, ids):
        # get subset dataframe
        subset_df = self.data_df.loc[ids]
        # perform subsampling
        subset_df = subset_df.reset_index().groupby(self.slide_var).apply(lambda x: self.tile_sampler(x))
        subset_df = subset_df.reset_index(drop=True).dropna(subset=[self.label_var])

        return subset_df

    
    def setup(self, stage=None):
        self.train_paths = self.subsample_tiles(self.train_ids)
        self.val_paths = self.subsample_tiles(self.val_ids)
        self.test_paths = self.subsample_tiles(self.test_ids)

        if torch.cuda.is_available():
            self.pin_memory = True
        else:
            self.pin_memory = False

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.train_dataset = SlideDataset(
                paths=self.train_paths.full_path.values,
                slide_ids=self.train_paths.index.values,
                labels=self.train_paths[self.label_var].values,
                transform_compose=self.train_transform
            )
            self.dev_dataset = SlideDataset(
                paths=self.val_paths.full_path.values,
                slide_ids=self.val_paths.index.values,
                labels=self.val_paths[self.label_var].values,
                transform_compose=self.eval_transform
            )

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test_dataset = SlideDataset(
                paths=self.test_paths.full_path.values,
                slide_ids=self.test_paths.index.values,
                labels=self.test_paths[self.label_var].values,
                transform_compose=self.eval_transform
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, pin_memory=self.pin_memory,
                          num_workers=self.workers)

    def val_dataloader(self):
        return DataLoader(self.dev_dataset, batch_size=self.batch_size, pin_memory=self.pin_memory,
                          num_workers=self.workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, pin_memory=self.pin_memory,
                          num_workers=self.workers)


# class SlideDataModule(pl.LightningDataModule):
#     def __init__(self, hparams):
#         super().__init__()
#     # def __init__(self, data_df, train_ids, val_ids, test_ids, train_transform, eval_transform,
#     #              label_var='angio_high', slide_var='slide_id',
#     #              tile_size=512, workers=8, batch_size=100, tiles_per_slide=500):
#     #     super().__init__()
#
#         self.dims = (3, self.hparams.hparams.tile_size, self.hparams.hparams.tile_size)
#
#     def tile_sampler(self, x):
#         samples = x.sample(min(len(x), self.hparams.tiles_per_slide))
#         return samples
#
#     def subsample_tiles(self, ids):
#         # get subset dataframe
#         subset_df = self.hparams.data_df.loc[ids]
#         # perform subsampling
#         subset_df = subset_df.reset_index().groupby(self.hparams.slide_var).apply(lambda x: self.hparams.tile_sampler(x))
#         subset_df = subset_df.reset_index(drop=True).dropna(subset=[self.hparams.label_var])
#
#         return subset_df
#
#     def prepare_data(self):
#         self.train_paths = self.hparams.subsample_tiles(self.hparams.train_ids)
#         self.val_paths = self.hparams.subsample_tiles(self.hparams.val_ids)
#         self.test_paths = self.hparams.subsample_tiles(self.hparams.test_ids)
#
#     def setup(self, stage=None):
#         if torch.cuda.is_available():
#             self.hparams.pin_memory = True
#         else:
#             self.hparams.pin_memory = False
#
#         # Assign train/val datasets for use in dataloaders
#         if stage == 'fit' or stage is None:
#             self.hparams.train_dataset = SlideDataset(
#                 paths=self.train_paths.full_path.values,
#                 slide_ids=self.train_paths.index.values,
#                 labels=self.train_paths[self.hparams.label_var].values,
#                 transform_compose=self.hparams.train_transform
#             )
#             self.hparams.dev_dataset = SlideDataset(
#                 paths=self.val_paths.full_path.values,
#                 slide_ids=self.val_paths.index.values,
#                 labels=self.val_paths[self.hparams.label_var].values,
#                 transform_compose=self.hparams.eval_transform
#             )
#
#         # Assign test dataset for use in dataloader(s)
#         if stage == 'test' or stage is None:
#             self.hparams.test_dataset = SlideDataset(
#                 paths=self.test_paths.full_path.values,
#                 slide_ids=self.test_paths.index.values,
#                 labels=self.test_paths[self.hparams.label_var].values,
#                 transform_compose=self.hparams.eval_transform
#             )
#
#     def train_dataloader(self):
#         return DataLoader(self.hparams.train_dataset, batch_size=self.hparams.batch_size, pin_memory=self.hparams.pin_memory,
#                           num_workers=self.hparams.workers)
#
#     def val_dataloader(self):
#         return DataLoader(self.hparams.dev_dataset, batch_size=self.hparams.batch_size, pin_memory=self.hparams.pin_memory,
#                           num_workers=self.hparams.workers)
#
#     def test_dataloader(self):
#         return DataLoader(self.hparams.test_dataset, batch_size=self.hparams.batch_size, pin_memory=self.hparams.pin_memory,
#                           num_workers=self.hparams.workers)
#

    
