import pytorch_lightning as pl
import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision.models as models
from argparse import ArgumentParser
import numpy as np
import pytorch_lightning as pl
from pl_bolts.models.self_supervised import SimCLR
from pl_bolts.models.self_supervised.resnets import resnet50_bn

class SimCLR_multiplex(SimCLR):
    """
    Args:
        batch_size: the batch size
        num_samples: num samples in the dataset
        warmup_epochs: epochs to warmup the lr for
        lr: the optimizer learning rate
        opt_weight_decay: the optimizer weight decay
        loss_temperature: the loss temperature
    """

    def __init__(self, num_input_channels, **kwargs):
        super().__init__(**kwargs)

        self.num_input_channels = num_input_channels
        self.encoder = self.init_encoder_mod()

    # separate fn since we'll do the default init first which needs to call `self.init_encoder` from base `SimCLR`
    def init_encoder_mod(self):
        encoder = resnet50_bn(return_all_feature_maps=False)

        # when using cifar10, replace the first conv so image doesn't shrink away
        encoder.conv1 = nn.Conv2d(
            self.num_input_channels, 64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        return encoder

    # TODO consider adding additional train/validation step logging metrics

