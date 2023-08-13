import torch
import torch.nn as nn
import torchvision
import torch.distributions as ds
from torchvision import transforms
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from glob import glob
import os


def create_sequential_block(layer_params, input_size, input_channels, encoder=True, residual=False):
    """
    Creates encoder/decoder convolutional sequential module container for VAE models

    Iterates through `layer_param_dict`, which specifies nn.Conv2d parameters
    (1) creates OrderedDict object whose entries correspond to filter number and a nn.Conv2d module object
    (2) wraps output of (1) in a nn.Sequential container
    (3) calculates output_size of block
    Returns: (nn.Sequential container, output_size)
    """
    # construct nn.Sequential container
    module_agg = []
    for param_dict in layer_params:
        if not residual:
            if encoder:
                conv_module = BasicConv2d_LRA(**param_dict)
            if not encoder:
                conv_module = BasicConvTranspose2d_LRA(**param_dict)
        if residual:
            if encoder:
                conv_module = BasicConv2d_LRA_residual(**param_dict)
            if not encoder:
                raise NotImplementedError
#                 conv_module = BasicConvTranspose2d_LRA(**param_dict)
        module_agg.append(conv_module)
    
    sequential = nn.Sequential(*module_agg)  # simpler to unpack than manually create dict!

    # calculate output_size using a fake input (instead of manually calculating)
    with torch.no_grad():
        dummy_input = torch.ones((1, input_channels, input_size, input_size))
        dummy_output = sequential(dummy_input)
        output_size = dummy_output.shape[-1]

    return sequential, output_size



## forward loss version
class MultitaskPretrainedModel(nn.Module):
    def __init__(self, encoder, 
                 input_channels=3, 
                 task0_classes=2, 
                 task1_classes=2,
                 encoding_channels=2048, 
                 dropout_prob=0.5,
                ):
        super().__init__()
        
        self.input_channels = input_channels
        self.task0_classes = task0_classes
        self.task1_classes = task1_classes
        self.encoding_channels = encoding_channels
        self.dropout = nn.Dropout(p=dropout_prob)

        self.condensation_stack = encoder

        # TODO consider generalizing to M tasks 
        self.classifier0 = nn.Sequential(
            nn.Linear(self.encoding_channels, self.task0_classes),
        )
        self.classifier1 = nn.Sequential(
            nn.Linear(self.encoding_channels, self.task1_classes),
        )
        
        self.loss = nn.CrossEntropyLoss()

        
    def forward(self, x, task0_labels, task1_labels):
        # downsample
        encoding = self.condensation_stack(x)
        encoding = self.dropout(encoding)

        batch_size = x.shape[0]
        encoding = encoding.view(batch_size,self.encoding_channels) 
        
        task0_logits = self.classifier0(encoding)
        task1_logits = self.classifier1(encoding)
        
        task0_loss = self.loss(task0_logits, task0_labels.long())
        task1_loss = self.loss(task1_logits, task1_labels.long())

        loss = task0_loss + task1_loss  # TODO consider scaling terms as option for each loss
        
        return (task0_logits, task1_logits), loss.unsqueeze(0)

class BasicConv2d_LRA(nn.Module):
    """
    from PyTorch's Inceptionv3 implementation
    https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py#L344
    """
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d_LRA, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True) # relu instead of leaky_relu this time
  

# to match BasicConv2d_LRA
class BasicConvTranspose2d_LRA(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConvTranspose2d_LRA, self).__init__()
        self.convt = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.convt(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
    

class BasicConv2d_LRA_residual(nn.Module):
    """
    from PyTorch's Inceptionv3 implementation
    https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py#L344
    """
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        out = self.conv(x)
        out += x  # residual connection via addition
        out = self.bn(out)
        return F.relu(out, inplace=True)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



class spatialAttention(nn.Module):
    """
    Encoder still assumed to produce a field of latents to attend over for prediction [likely a frozen pretrained prepool resnet50]
    Secondary set of layers for interaction, upsampling, attention build on top of input field of latents
    """
    def __init__(self, 
                 encoder, 
                 conv_stack,
                 upsampling_stack,
                 input_channels=3, 
                 num_classes=2, 
                 attention_size=128, 
                 encoding_channels=256, 
                 dropout_prob=0.5):
        super().__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.encoding_channels = encoding_channels

        # misc functions
        self.LRA = nn.LeakyReLU(inplace=True)
        self.batchnorm1d_mlp = nn.BatchNorm1d(num_features=self.encoding_channels)  # self.encoding_channels hardcoded
        self.dropout = nn.Dropout(p=dropout_prob)

        self.encoder = encoder
        self.conv = conv_stack
        self.upsample = upsampling_stack

        
        self.attention_size = attention_size

        self.gate = nn.Sequential(
            nn.Linear(self.encoding_channels, self.attention_size),  # denoted as `U` in paper eq (9)
            nn.Sigmoid()
        )

        self.attention = nn.Sequential(
            nn.Linear(self.encoding_channels, self.attention_size),  # denoted as `V` in paper eq (9)
            nn.Tanh()
        )

        self.attention_agg = nn.Linear(self.attention_size, 1)  # denoted as `w` in paper eq (9)

        self.classifier = nn.Sequential(
            nn.Linear(self.encoding_channels, self.num_classes),
        )
        
        self.loss = nn.CrossEntropyLoss()
        
        
    def run_spatial_attention(self, x):
        batch_size = len(x)
        channels = x.shape[1] 
#         encoding = encoding.view(batch_size, -1, self.encoding_channels) 
        encoding = x.view(batch_size, -1, channels) 

        gate_weights = self.gate(encoding) # B x unrolled_size x attention_size
        ungated_attention_encoding = self.attention(encoding) # B x unrolled_size x attention_size

        gated_attention_encoding = gate_weights * ungated_attention_encoding  # B x unrolled_size x attention_size
        
        gated_attention_logits = self.attention_agg(gated_attention_encoding)  # B x unrolled_size x 1
        gated_attention_logits = gated_attention_logits.permute(0,2,1)
        
        attention = F.softmax(gated_attention_logits, dim=-1)  # B x 1 x unrolled_size
        attention_field_size = int(attention.shape[-1] ** 0.5)
        attention_reshape = attention.view(batch_size, attention_field_size, attention_field_size)
        
        aggregate_encoding = torch.bmm(attention, encoding)  # (B x 1 x unrolled_size) @ (B x unrolled x encoding_channels) = (B x 1 x encoding_channels)
        aggregate_encoding = torch.squeeze(aggregate_encoding, 1)  # push to (B x encoding_channels)
        
        return aggregate_encoding, attention_reshape

    
    def forward(self, x, labels):
        # downsample
        encoding = self.encoder(x)
        encoding = self.dropout(encoding)
#         print(encoding.shape)
        
        encoding = self.conv(encoding)  # interaction
        encoding = self.dropout(encoding)
#         print(encoding.shape)

        encoding = self.upsample(encoding) 
        encoding = self.dropout(encoding)
#         print(encoding.shape)

        aggregate_encoding, attention = self.run_spatial_attention(encoding)
        
        pred_logits = self.classifier(aggregate_encoding)  # B x num_classes
        loss = self.loss(pred_logits, labels.long())

        return (pred_logits, attention), loss.unsqueeze(0)
