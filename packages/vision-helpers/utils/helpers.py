# import torch
# import torchvision
# import torch.distributions as ds
# from torchvision import transforms
# import torch.optim as optim
# import torch.nn.functional as F
# import torch.nn as nn

import seaborn as sns

from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from glob import glob
import os
from sklearn.decomposition import PCA
import umap
import datetime
import sys
import tqdm
from torch import sparse
import PIL
from PIL import Image, ImageDraw
import json
import xml.etree.ElementTree as ET
from tqdm import tqdm, tqdm_notebook

from sklearn.neighbors import NearestNeighbors
import openensembles as oe
import lifelines
from lifelines import CoxPHFitter, KaplanMeierFitter

import scanpy as sc
sc.settings.verbosity = 3 
import anndata



def get_merged_df(df1, df2, drop_col=None):
    merge = df1.merge(df2, how='left', left_index=True, right_index=True)
    if drop_col is not None:
        merge = merge.dropna(subset=[drop_col])
    return merge

def set_rc(x=10,y=10, font_scale=1.):
    sns.set(rc={'figure.figsize':[x,y]}, font_scale=font_scale)
    
def get_ecdf(series, interval_step, interval_min=0, interval_max=1.,):
    intervals = np.arange(interval_min, interval_max+interval_step, interval_step)
    ecdf = [(series <= x).mean() for x in intervals]
    plt.plot(intervals,ecdf)
    return intervals, ecdf

def get_indices(series):
    return [idx for idx, x in series.iteritems() if x]

def reset_set_idx(df, idx_name):
    return df.reset_index().set_index(idx_name)

def run_embedding(encoding_array, annotation_df, n_pcs=50, n_neighbors=25, use_rapids=True, run_louvain=False):
    print('creating AnnData object...')
    adata = anndata.AnnData(encoding_array)
    adata.obs = annotation_df # annotate data
    if use_rapids:
        print('using rapids GPU implementation')
        method='rapids'
        flavor='rapids'
    else:
        method='umap'
        flavor='vtraag'
    # run pca
    if n_pcs is not None:
        sc.tl.pca(adata, svd_solver='arpack', n_comps=n_pcs)
        # get neighbor graph
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, method=method)
    else:
        # get neighbor graph
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, use_rep='X', method=method)
    # get umap embedding
#     sc.tl.umap(adata, method=method)
    sc.tl.umap(adata)

    if run_louvain:
        # run louvain clustering
        sc.tl.louvain(adata, flavor=flavor)
    return adata