import torch
from torch import nn, optim
from torchvision import datasets, models, transforms
from collections import OrderedDict
import argparse
import os
import sys
import numpy as np
import configparser
sys.path.append('/home/nyman/')
sys.path.append('/home/nyman/tmb_bot/')

# self module
import model_init
from spatial_attention_model_specification_resnet50 import spatialAttentionSingleEncoder

# helper for creating train/eval transforms
def create_transforms(full_size, crop_size):
    train_transform = transforms.Compose(
        [
            transforms.Resize(full_size),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=32. / 255.,
                contrast=0.5,
                saturation=0.5,
                hue=0.1
            ),
            transforms.ToTensor(),
            # TODO decide if mean/std here should be args as well
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dev_transform = transforms.Compose(
        [
            transforms.Resize(full_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return train_transform, dev_transform



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', '-o', type=str, default='./')
    parser.add_argument('--model_filename', '-mf', type=str, default='saved_initialized_models.pth')
    parser.add_argument('--transform_filename', '-tf', type=str, default='saved_transforms.pth')
    parser.add_argument('--seed', type=int, default=None, metavar='N',
                        help='set a random seed for torch and numpy (default: None)')
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print('using random seed {} during model initialization'.format(args.seed))
    else:
        print('NOT fixing random seed during model initialization')

    #### Initialize and store models in list for exporting
    NUM_CLASSES = 2
    FULL_SIZE = 512

    model_agg = []
    all_transforms = []
    config = configparser.ConfigParser()  # config file to write non object info to


    #### model 0 ####
    architecture = 'resnet18'
    FEATURE_EXTRACT = False
    # create model and aggregate
    model_ft, CROP_SIZE = model_init.initialize_model(architecture, NUM_CLASSES, feature_extract=FEATURE_EXTRACT,
                                                      use_pretrained=True)
    print(CROP_SIZE)
    model_agg.append(model_ft)
    all_transforms.append(create_transforms(FULL_SIZE, CROP_SIZE))
    # write to config INI
    config[0] = {
        'forward_loss': False,
        'lr': 1e-4,
    }

    #### model 1 ####
    architecture = 'resnet50'
    FEATURE_EXTRACT = False
    # create model and aggregate
    model_ft, CROP_SIZE = model_init.initialize_model(architecture, NUM_CLASSES, feature_extract=FEATURE_EXTRACT,
                                                      use_pretrained=True)
    model_agg.append(model_ft)
    print(CROP_SIZE)
    all_transforms.append(create_transforms(FULL_SIZE, CROP_SIZE))
    # write to config INI
    config[1] = {
        'forward_loss': False,
        'lr': 1e-4,
    }

    #### model 2 example: spatial attention with pretrained resnet50 ####
    CROP_SIZE = 224
    ATTENTION_SIZE = 128
    # grab pre avgpooling layers in resnet50 model
    FROZEN_PRETRAINED = False
    image_modules = list(models.resnet50(pretrained=True).children())[:-2]
    resnet_prepool = nn.Sequential(*image_modules)

    # optionally, freeze resnet pretrained layers before adding to model
    if FROZEN_PRETRAINED:
        for param in resnet_prepool.parameters():
            param.requires_grad = False

    # create model and aggregate
    model = spatialAttentionSingleEncoder(
        encoder=resnet_prepool,
        encoding_channels=2048,  # resnet50 hardcoded
        attention_size=ATTENTION_SIZE,
    )
    model_agg.append(model)
    print(CROP_SIZE)
    all_transforms.append(create_transforms(FULL_SIZE, CROP_SIZE))
    # write to config INI
    config[2] = {
        'forward_loss': True,
        'lr': 1e-4,
    }


    # collect all models
    all_models = OrderedDict({idx: model for idx, model in enumerate(model_agg)})

    # save models to file
    out_path = os.path.join(args.out, args.model_filename)
    torch.save(all_models, out_path)

    out_path = os.path.join(args.out, args.transform_filename)
    torch.save(all_transforms, out_path)

    out_path = os.path.join(args.out, 'model_config.ini')
    with open(out_path, 'w') as configfile:
        config.write(configfile)
