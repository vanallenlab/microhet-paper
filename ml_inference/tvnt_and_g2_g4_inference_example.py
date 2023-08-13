import torch
import torchvision
import torch.distributions as ds
from torchvision import transforms
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import os
from sklearn.decomposition import PCA
import datetime
import sys
import tqdm
from torch import sparse
import PIL
from PIL import Image, ImageDraw
import json
import xml.etree.ElementTree as ET
from tqdm import tqdm, tqdm_notebook
from PIL import Image

# set up logging
# sys.stderr = open('20201007_TvNT_inference_stderr.txt', 'w')
# sys.stdout = open('20201007_TvNT_inference_stdout.txt', 'w')

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class SlideDatasetMod(data.Dataset):
    """
    Modification of vanilla `tmb_bot.utilities.Dataset` class to facilitate having
    a label for classification as well as the slide name itself
    """
    def __init__(self, paths, x, y, slide_ids, transform_compose):
        """
        Paths and labels should be array like
        x: xcoord of tile
        y: ycoord of tile
        """
        self.paths = paths
        self.slide_ids = slide_ids
        self.x = x
        self.y = y 
        self.transform = transform_compose

    def __len__(self):
        return self.paths.shape[0]

    def __getitem__(self, index):
        img_path = self.paths[index]
        pil_file = pil_loader(img_path)
        pil_file = self.transform(pil_file)
        slide_id = self.slide_ids[index]
        x_coord = self.x[index]
        y_coord = self.y[index]

        return pil_file, x_coord, y_coord, slide_id

# mc_lightning
from mc_lightning.models.resnet.resnet_module import PretrainedResnet50FT
from mc_lightning.models.resnet.resnet_transforms import RGBTrainTransform, RGBEvalTransform
from mc_lightning.legacy import model_init

# puling from model-comparison utilities.py
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score


def get_checkpoint_and_data_pl(output_dir, paths_file, fold_idx, model=PretrainedResnet50FT, version_idx=0, epoch=9, tile_size=512, crop_size=224):
    """
    PL version of model-comparison utilities get_checkpoint_and_data function
    """
    paths_df = pd.read_pickle(paths_file)

    checkpoint_path =  os.path.join(output_dir, f'fold{fold_idx}/lightning_logs/version_{version_idx}/checkpoints/*.ckpt')
    checkpoint_path = glob(checkpoint_path)[-1]

    val_ids = os.path.join(output_dir, f'20x_512px_fold{fold_idx}_val_slide_ids.csv')
    val_ids = pd.read_csv(val_ids).iloc[:, 1].values
    val_paths_df = paths_df.loc[val_ids]

    loaded_model = model.load_from_checkpoint(checkpoint_path)
    loaded_model.eval()

    eval_transform = RGBEvalTransform(tile_size, crop_size)

    return loaded_model, eval_transform, val_paths_df

def get_checkpoint_and_data_pl_mod(output_dir, paths_file, fold_idx, model=PretrainedResnet50FT, version_idx=0, epoch=9, tile_size=512, crop_size=224):
    """
    PL version of model-comparison utilities get_checkpoint_and_data function
    slight variation of base function that expects output_dir to have fold info already in it 
    """
    paths_df = pd.read_pickle(paths_file)

    checkpoint_path =  os.path.join(output_dir, f'lightning_logs/version_{version_idx}/checkpoints/*.ckpt')
    checkpoint_path = glob(checkpoint_path)[-1]

    val_ids = os.path.join(output_dir, f'20x_512px_fold{fold_idx}_val_slide_ids.csv')
    val_ids = pd.read_csv(val_ids).iloc[:, 1].values
    val_paths_df = paths_df.loc[val_ids]

    loaded_model = model.load_from_checkpoint(checkpoint_path)
    loaded_model.eval()

    eval_transform = RGBEvalTransform(tile_size, crop_size)

    return loaded_model, eval_transform, val_paths_df


######################################
# load tumor vs nontumor model (based on older MC code)
#### 20200610_model_spec_tumor_vs_nontumor_0.py model 1 ####
architecture = 'resnet50'
FEATURE_EXTRACT = False
NUM_CLASSES = 2
# create model and aggregate
tvnt_model, CROP_SIZE = model_init.initialize_model(architecture, NUM_CLASSES, feature_extract=FEATURE_EXTRACT,
                                                  use_pretrained=True)

# LOAD STATE DICT
state_dict_path = '/mnt/disks/bms/model_checkpoints/profile_manual_tumor_vs_nontumor_statedict_fold1_model1_002.pth'
tvnt_model.load_state_dict(torch.load(state_dict_path))
tvnt_model = nn.DataParallel(tvnt_model).to('cuda')
tvnt_model.eval()
######################################


print('Analyzing outputs from `20210312_g2_g4_aug_strength_sweep_higher_bs.sh` runner on profile ccrcc data')
LABEL_NAME = 'g4_not_g2'
NUM_CLASSES = 2
# SUBSAMPLE_TPS = 100

BATCH_SIZE = 600
# BATCH_SIZE = 200
WORKERS = 8
CROP_SIZE = 224
        
checkpoint_map = {
    0:'/mnt/disks/bms/model_checkpoints/20210312_g2_g4_checkpoints_grab/fold0_aug_strength_4.0_tps1000_epoch=3.ckpt',
    1:'/mnt/disks/bms/model_checkpoints/20210312_g2_g4_checkpoints_grab/fold1_aug_strength_4.0_tps1000_epoch=4.ckpt',
    2:'/mnt/disks/bms/model_checkpoints/20210312_g2_g4_checkpoints_grab/fold2_aug_strength_4.0_tps1000_epoch=3.ckpt',
    3:'/mnt/disks/bms/model_checkpoints/20210312_g2_g4_checkpoints_grab/fold3_aug_strength_4.0_tps1000_epoch=4.ckpt',
}

model_map = {}
for key, checkpoint_path in checkpoint_map.items():
    eval_transform = RGBEvalTransform(512, CROP_SIZE)
    model = PretrainedResnet50FT.load_from_checkpoint(checkpoint_path)
    model = nn.DataParallel(model).to('cuda')
    model.eval()
    model_map[key] = model

# add TvNT model too
model_map['tvnt'] = tvnt_model


print('Using previous inference data to avoid rerunning tumor vs non-tumor filtering twice')
dataset_map = {
    'cm025':pd.read_csv('/mnt/disks/manual_bms_tiles/20210426_tile_paths_unprocessed_manual_cm025_635_slides.csv'),
}

print('Saving per-slide DFs to avoid memory issues')
existing_ids = []

# print('Only running on missing KIRC data')
# partial_kirc_updated_inf =  pd.read_csv('/home/nyman/20210315_temp_df_agg_backup_ALL_TILES.csv')
# existing_ids = partial_kirc_updated_inf.slide_id.unique()[:-1] # assume last ID was interrupted mid-run
# print(f'Found {len(existing_ids)} existing IDs')
# # print(f'RUNNING PARTIAL INFERENCE ON {SUBSAMPLE_TPS} TPS')

# dataset_subset_map = {}
# for dataset_label, dataset_df in dataset_map.items():
#     subset_df = dataset_df.groupby('slide_id').apply(lambda x: x.sample(min(len(x), SUBSAMPLE_TPS))).reset_index(drop=True)
#     dataset_subset_map[dataset_label] = subset_df

#####################################
for dataset_label, samples_df in dataset_map.items():
    for slide_id in tqdm(samples_df.slide_id.unique()):
        if slide_id not in existing_ids:
            try:
                df_agg = []
                df_subset = samples_df.loc[samples_df.slide_id == slide_id].set_index(['x','y'])

                # CREATE DATASET
                temp_dataset = SlideDatasetMod(
                    paths = df_subset.full_path.values,
                    x = df_subset.reset_index().x.values,
                    y = df_subset.reset_index().y.values,
                    slide_ids = df_subset.slide_id.values,
                    transform_compose = eval_transform,
                )

                # CREATE DATALOADER
                temp_dataloader = DataLoader(
                    dataset=temp_dataset, 
                    batch_size=BATCH_SIZE, 
                    num_workers=WORKERS,
                    shuffle=False
                )

                # RUN INFERENCE
                with torch.no_grad():
                    for batch_idx, (imgs, x_coords, y_coords, slide_ids) in enumerate(temp_dataloader):
                        for fold_idx, model in model_map.items():
                            print(f'Model {fold_idx}, running inference on {slide_id} ...')
                            df_subset_clone = df_subset.copy()  

                            model.eval()
                            out = model(imgs)
                            prob = out.softmax(-1) # keep as multinomial prob
                            for tile_idx in range(len(out)):
                                # UPDATE PATHS_DF WITH p(label) OUTPUTS
                                for class_idx in range(NUM_CLASSES):
                                    df_subset_clone.loc[(x_coords[tile_idx].item(), y_coords[tile_idx].item()), f'class_{class_idx}_model_prob'] = prob[tile_idx, class_idx].cpu().numpy()

                            if fold_idx != 'tvnt':
                                df_subset_clone['model_id'] = f'profile_ccrcc_fold{fold_idx}_20210312'
                            else:
                                df_subset_clone['model_id'] = 'tvnt'
                            df_subset_clone['dataset'] = dataset_label

                            # store updated paths_df subset
                            df_agg.append(df_subset_clone.dropna())

                # save each slide's inference set separately to avoid memory issues (probably killed first run since 512x4 sets of predictions lol)
                pd.concat(df_agg).to_csv(f'./20210427_manual_cm025_inference/20210427_manual_cm025_inference_using_profile_g2_g4_models__{slide_id}.csv')


            except Exception as e:
                print(e)
                print('Issue with {}'.format(slide_id))
        else:
            print(f'Already processed {slide_id}')

