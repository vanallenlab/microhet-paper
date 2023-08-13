import skimage

import os
os.environ["JAVA_HOME"] = "/opt/conda/envs/pathml"
# import pathml
# from pathml.core.slide_classes import HESlide

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
from collections import defaultdict
from glob import glob
from matplotlib.patches import Rectangle
from copy import deepcopy

sys.path.append('/home/jupyter/vision-helpers/')
sys.path.append('/home/jupyter/image-graphs')

# vision-helpers imports
from nuclei_processing import run_contour_calling_three_class
from utils.helpers import set_rc, get_merged_df, reset_set_idx, get_indices
from utils.spatialclustering import plot_assignment_clusters_combined, assign_threeway_category
from utils.lifelines_helpers import run_cph_comparison, run_cph_comparison_multivar, extract_cph_results

# lifelines
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.plotting import plot_lifetimes


# sklearn/skimage/scipy
from sklearn.preprocessing import StandardScaler
from skimage.transform import rescale, resize, downscale_local_mean
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import f1_score
import scipy as sp
from scipy import ndimage as ndi
from skimage.morphology import disk
from skimage.future.graph import rag_mean_color, show_rag
from skimage.segmentation import watershed, expand_labels
from skimage import data
from skimage.filters import rank
from skimage.util import img_as_ubyte
from skimage import data, io, segmentation, color
from skimage.future import graph

# pytorch
import torch
import torch.utils.data as data
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from PIL import Image
# from mc_lightning.utilities.utilities import pil_loader

# dask
import dask
from dask.distributed import Client, LocalCluster

import networkx as nx

def set_rc(x=10,y=10, font_scale=1., style='white', font='Helvetica', **kwargs):
    sns.set(rc={'figure.figsize':[x,y]}, font_scale=font_scale, style=style, font=font, **kwargs)
    
set_rc()


### global
HUE_ORDER = ['stroma','pred_g2','intermediate_grade','pred_g4']
# MIN_SEGMENT_SIZE = 50
# NODE_DIFF_CUTOFF = 0.25
# MIN_TIL_COUNT = 10
# TIL_ISO_CUTOFF = 10
# TIL_HIGH_CUTOFF = 25
# TIL_AREA_CUTOFF = 50
# FRAC_CUTOFF = 0.2

#######################################
def map_qbins_to_label(series, n_bins):
    if n_bins == 3:
        return series.map({'quantile_bin_0':'Lower Grade', 'quantile_bin_1':'Intermed. Grade', 'quantile_bin_2':'Higher Grade'})
    else:
        return series.map({'quantile_bin_0':'Lower Grade', 'quantile_bin_1':'Higher Grade'})
    
def map_bool_to_label(series, label_a='no', label_b='yes', prefix='series'):
    """
    Takes a bool series and uses its column name to create labels
    """
    if prefix == 'series':
        prefix = series.name
        return series.map({False:f'{prefix}_{label_a}', True:f'{prefix}_{label_b}'})
    else:
        return series.map({False:f'{label_a}', True:f'{label_b}'})

def move_leg(sns_plot, x=1.25, y=0.5):
    sns_plot.legend(loc='center left', bbox_to_anchor=(x, y), ncol=1)    

def get_qbins(df, col, num_bins):
    bin_series, bins = pd.qcut(df[col], retbins=True, q=np.linspace(0,1,num_bins+1), labels=[f'quantile_bin_{x}' for x in range(num_bins)])
    return bin_series, bins

def get_pbins(df, col, num_bins):
    bin_series, bins = pd.cut(df[col], retbins=True, bins=np.linspace(0,1,num_bins+1), labels=[f'prob_bin_{x}' for x in range(num_bins)])
    return bin_series, bins

## updated in 20210408NB
def plot_assignment_clusters_combined(clusters_subset, x_var='x', y_var='y', size=6, use_style=False, hue_order=HUE_ORDER, ax=None, marker_size=5, 
                                      legend='auto', legend_out=True, invert_yaxis=False):
    """
    Assumes that we will rescale the x,y distance to be [0,1]
    """
#     set_rc(size,size)
    if use_style:
        g = sns.scatterplot(x_var,y_var, data=clusters_subset, hue='meta',style='meta_and_label', hue_order=hue_order, ax=ax, s=marker_size, legend=legend)
    else:
        g = sns.scatterplot(x_var,y_var, data=clusters_subset, hue='meta', hue_order=hue_order, ax=ax, s=marker_size, legend=legend)

    if legend_out & (legend != False):
        g.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)
    if invert_yaxis:
        g.invert_yaxis()
    return g

def plot_multiple_assignment_groups(metrics, assignments, cohort, n_samples=5, sharexy=True, g2_cutoff=0.10, g4_cutoff=0.10, 
                                    plot_size=3, marker_size=10, meta_vars=['stroma', 'pred_g2', 'intermediate_grade', 'pred_g4']):
    set_rc(n_samples*plot_size, plot_size, font_scale=2)

    fig, axes = plt.subplots(1,n_samples, sharex=sharexy, sharey=sharexy)
    crit = (metrics.cohort == cohort) & (metrics.candidate_category_ext == 'homogeneous') 
    for rel, idx in enumerate(metrics.loc[crit].sample(n_samples).index.values):
    #     leg_out = False if rel != n_samples else True
        if rel == (n_samples-1):
            leg = True
        else:
            leg = False
        plot_assignment_clusters_combined(assignments.loc[idx], use_style=False, marker_size=marker_size, ax=axes[rel], hue_order=meta_vars, legend=leg, legend_out=True)
    plt.suptitle(f'{cohort.upper()}, Homogeneous')
    plt.show()

    fig, axes = plt.subplots(1,n_samples, sharex=sharexy, sharey=sharexy)
    crit = (metrics.cohort == cohort) & (metrics.candidate_category_ext == 'heterogeneous_mincount') 
    for rel, idx in enumerate(metrics.loc[crit].sample(n_samples).index.values):
    #     leg_out = False if rel != n_samples else True
        if rel == (n_samples-1):
            leg = True
        else:
            leg = False
        plot_assignment_clusters_combined(assignments.loc[idx], use_style=False, marker_size=marker_size, ax=axes[rel], hue_order=meta_vars, legend=leg, legend_out=True)
    plt.suptitle(f'{cohort.upper()}, Heterogeneous ({g2_cutoff*100}/{g4_cutoff*100}% min G2/G4 presence)')
    plt.show()
    
    fig, axes = plt.subplots(1,n_samples, sharex=sharexy, sharey=sharexy)
    crit = (metrics.cohort == cohort) & (metrics.candidate_category_ext == 'heterogeneous_other') 
    for rel, idx in enumerate(metrics.loc[crit].sample(n_samples).index.values):
    #     leg_out = False if rel != n_samples else True
        if rel == (n_samples-1):
            leg = True
        else:
            leg = False
        plot_assignment_clusters_combined(assignments.loc[idx], use_style=False, marker_size=marker_size, ax=axes[rel], hue_order=meta_vars, legend=leg, legend_out=True)
    plt.suptitle(f'{cohort.upper()}, Heterogeneous (other)')
    plt.show()

def add_ws_coords(df_agg):
    df_agg['ws_x'] = (df_agg['tx']*512) + df_agg['cx']
    df_agg['ws_y'] = (df_agg['ty']*512) + df_agg['cy']

def filter_results(df):
    return df.loc[(df['cx'] >= 0) & (df['cy'] >= 0) & (df['cx'] <= 512) & (df['cy'] <= 512)]



def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
    
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

    def __len__(self):
        'Denotes the total number of samples'
        return self.paths.shape[0]

    def __getitem__(self, index):
        'Generates one sample of data'
        img_path = self.paths[index]
        pil_file = pil_loader(img_path)
        pil_file = self.transform(pil_file)
        slide_id = self.slide_ids[index]
        label = self.labels[index]

        return pil_file, label, slide_id


class RGBEvalTransform(object):
    def __init__(self, full_size, crop_size, norm_mean=[0.485, 0.456, 0.406], norm_std=[0.229, 0.224, 0.225]):
        self.full_size = full_size
        self.crop_size = crop_size
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        normalizer = transforms.Normalize(mean=self.norm_mean, std=self.norm_std)
        data_transforms = transforms.Compose([transforms.Resize(self.full_size),
                                              transforms.CenterCrop(self.crop_size),
                                              transforms.ToTensor(),
                                              normalizer])
        self.eval_transform = data_transforms

    def __call__(self, sample):
        transform = self.eval_transform
        out = transform(sample)
        return out

### Formalize smoothing procedure 

def run_inference_smoothing(df, dist_type='uniform', n_neighbors=4, tumor_cutoff=0.5, lower_grade_cutoff=1/3, upper_grade_cutoff=2/3):
    """
    stage 1: Smooth all p(Tumor) values 
    stage 2: select rough estimate of where tumor data is via `tumor_cutoff`
    stage 3: smooth pre-segmentation estimate for G4 vs G2 info  
    
    Stage 2 is needed even though imperfect, since otherwise the Nearest Neighbor smoothing might dip into 
    the regions of non-tumor that can have strange p(G4 not G2) behavior 
    """

    # first smooth p(Tumor) since it will be used subsequently
    col_type = 'tumor'
    knr = KNeighborsRegressor(n_neighbors=n_neighbors, weights=dist_type)
    knr.fit(df[['x','y']].values,  df[[f'prob_{col_type}']].values)
    smoothed_preds = knr.predict(df[['x','y']].values)
    df[f'smoothed_prob_{col_type}'] = smoothed_preds

    # now subset on putative tumor regions and smooth p(G4 not G2)
    col_type = 'g4_not_g2'
    filtered_df = df.loc[df['smoothed_prob_tumor'] > tumor_cutoff]
    knr = KNeighborsRegressor(n_neighbors=n_neighbors, weights=dist_type)
    knr.fit(filtered_df[['x','y']].values,  filtered_df[[f'prob_{col_type}']].values)
    smoothed_preds = knr.predict(df[['x','y']].values)
    df[f'smoothed_prob_{col_type}'] = smoothed_preds
    
    # now bin according to smoothed values
    df = assign_threeway_category(df, 
                                target_var='smoothed_prob_g4_not_g2', tumor_var='smoothed_prob_tumor',
                                tumor_cutoff=tumor_cutoff,
                                lower_var_cutoff=lower_grade_cutoff, upper_var_cutoff=upper_grade_cutoff)
    return df

### Formalize nuclei post-processing (ratio comparisons, etc)

def get_stroma_tumor_ratio(df):
    try:
        counts = df.tissue_class.value_counts()
        ratio = counts['stroma'] / counts['tumor']
    except:
        ratio = np.nan
    return ratio

def get_til_tumor_ratio(df):
    try:
        counts = df.tissue_class.value_counts()
        ratio = counts['til'] / counts['tumor']
    except:
        ratio = np.nan
    return ratio

def get_cell_counts(df, tissue_class):
    try:
        counts = df.tissue_class.value_counts()[tissue_class]
    except:
        counts = np.nan
    return counts


# https://stackoverflow.com/questions/25579227/seaborn-implot-with-equation-and-r2-text
def annotate(data, x, y, **kws):
    r, p = sp.stats.pearsonr(data[x], data[y])
    ax = plt.gca()
    ax.text(.05, 0.01, 'r={:.2f}, p={:.2g}'.format(r, p),
            transform=ax.transAxes)

#### Formalize ROI checking prompt [Generalize from 20210422NB]

def get_example_info_kirc(unique_id, metrics, assignments):
    plotting_df = assignments.loc[unique_id]
    plotting_df['full_path'] = plotting_df['full_path'].apply(lambda x: x.replace('tcga-kidney-tiles','kirc'))
    
    seg_img_paths = glob(f'/mnt/disks/bms/hovernet_inference_outputs/kirc_subset_77/*{unique_id}*png')
    seg_img_paths = pd.DataFrame(seg_img_paths, columns=['seg_path'])
    seg_img_paths['x'] = seg_img_paths.seg_path.apply(lambda x: x.split('_')[-2]).astype(int)
    seg_img_paths['y'] = seg_img_paths.seg_path.apply(lambda x: x.split('_')[-1].split('.png')[0]).astype(int)
    plotting_df = get_merged_df(plotting_df,  seg_img_paths.set_index(['x','y'])).reset_index()
    plotting_df = run_inference_smoothing(plotting_df.reset_index(), lower_grade_cutoff=1/3, upper_grade_cutoff=2/3)
        
    return plotting_df


def get_heatmap(df, col, **kwargs):
    sns.heatmap(df[col].unstack().transpose(), **kwargs)

def get_heatmap_image(df, col, fill_val=np.nan):
    return df[col].unstack().transpose().fillna(fill_val).values

def reorient_probs(value):
    if value < 0.5:
        return 1-value
    else:
        return value

def get_separated_heatmaps(plotting_df, col='smoothed_prob_g4_not_g2'):
    x_max, y_max = plotting_df[['x','y']].max()
    
    placeholders = {}

    for meta, df in plotting_df.groupby('meta'):
        placeholder = np.ones((y_max+1, x_max+1))*-1
        for (x,y), row in df.set_index(['x','y']).iterrows():
            if row['meta'] == 'stroma':
                    placeholder[y,x] = np.nan
            else:
                placeholder[y,x] = row[col]

        placeholders[meta] = placeholder

    placeholder = np.ones((y_max+1, x_max+1))*-1
    for (x,y), row in plotting_df.set_index(['x','y']).iterrows():
        if row['meta'] == 'stroma':
                placeholder[y,x] = np.nan
        else:
            placeholder[y,x] = row[col]

    placeholders['all'] = placeholder


    placeholder = np.ones((y_max+1, x_max+1))*-1
    for (x,y), row in plotting_df.set_index(['x','y']).iterrows():
        if row['meta'] == 'stroma':
                placeholder[y,x] = -1
        else:
            placeholder[y,x] = row[col]

    placeholders['nonstroma'] = placeholder
    
    return placeholders

def run_watershed(plotting_df, denoise_disk_size=4, grad_disk_size=1, gradient_cutoff=45, include_reoriented_gradient=False):
    temp_df = plotting_df.copy()
    temp_df = temp_df.loc[temp_df.meta != 'stroma']

    image = get_heatmap_image(temp_df.set_index(['x','y']), 'smoothed_prob_g4_not_g2')
    reoriented_image = get_heatmap_image(temp_df.set_index(['x','y']), 'reoriented_smoothed_prob_g4_not_g2')

    mask = ~np.isnan(image)

    # denoise image
    denoised = rank.median(image, disk(DENOISE_DISK_SIZE), mask=mask)
    denoised_reoriented = rank.median(reoriented_image, disk(denoise_disk_size), mask=mask)

    # find continuous region (low gradient)
    marker_grad = rank.gradient(denoised, disk(5), mask=mask)
    markers = (marker_grad < gradient_cutoff) & mask
    markers = ndi.label(markers)[0]

    # get local gradient
    gradient = rank.gradient(denoised, disk(grad_disk_size), mask=mask)
    gradient_reoriented = rank.gradient(denoised_reoriented, disk(grad_disk_size), mask=mask)
    if include_reoriented_gradient:                  
        combined_local_gradient = gradient + gradient_reoriented
    else:
        combined_local_gradient = gradient
        
    # process the watershed
    labels = watershed(combined_local_gradient, markers, mask=mask)
    
    return labels, combined_local_gradient, image



def run_watershed_dict(image_dict, image_subset_id='nonstroma', smoothing_mode='median', denoise_disk_size=4, grad_disk_size=1, gradient_cutoff=45, use_markers=True):
    mask = image_dict[image_subset_id] != -1

    # denoise image
    if smoothing_mode == 'median':
        denoised = rank.median(image_dict[image_subset_id], disk(denoise_disk_size), mask=mask)
    if smoothing_mode == 'mean':
        denoised = rank.mean(image_dict[image_subset_id], disk(denoise_disk_size), mask=mask)
    if smoothing_mode == None:
        print('skipping denoising...')
        denoised = image_dict[image_subset_id]
    
    
    # get local gradient
    gradient = rank.gradient(denoised, disk(grad_disk_size), mask=mask)
    
    if use_markers:
        # find continuous region (low gradient)
        marker_grad = rank.gradient(denoised, disk(5), mask=mask)
        markers = (marker_grad < gradient_cutoff) & mask
        markers = ndi.label(markers)[0]
    else:
        # use default local minima behavioro
        markers = None
        
    # process the watershed
    labels = watershed(gradient, markers, mask=mask)

    return {'labels':labels, 'gradient':gradient, 'markers':markers}

def quad_plot(plotting_df, labels, nonstroma_image, full_image, gradient, mark_boundaries=False, plotting_mode='overlay'):
    fig, axes = plt.subplots(2,2)

    plot_assignment_clusters_combined(plotting_df, use_style=False, marker_size=25, ax=axes[0,1], invert_yaxis=True)
    axes[0,0].imshow(labels, cmap=plt.cm.nipy_spectral, alpha=1)

    out = color.label2rgb(labels, nonstroma_image, kind=plotting_mode, bg_label=0)
    if mark_boundaries:
        out = segmentation.mark_boundaries(out, labels, (0, 0, 0))
    axes[1,1].imshow(full_image != -1)
    axes[1,1].imshow(out, alpha=0.6)

    axes[1,0].imshow(out, alpha=0.6)
    axes[1,0].imshow(gradient, cmap=plt.cm.nipy_spectral, alpha=0.6)
    
    return fig

from skimage import data, io, segmentation, color
from skimage.future import graph
import numpy as np


def _weight_mean_color(graph, src, dst, n):
    """Callback to handle merging nodes by recomputing mean color.

    The method expects that the mean color of `dst` is already computed.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    data : dict
        A dictionary with the `"weight"` attribute set as the absolute
        difference of the mean color between node `dst` and `n`.
    """

    diff = graph.nodes[dst]['mean color'] - graph.nodes[n]['mean color']
    weight = np.linalg.norm(diff)
#     return {'weight': weight, 'diff':diff}
    return {'weight': weight}


def merge_mean_color(graph, src, dst):
    """Callback called before merging two nodes of a mean color distance graph.

    This method computes the mean color of `dst`.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    """
    graph.nodes[dst]['total color'] += graph.nodes[src]['total color']
    graph.nodes[dst]['pixel count'] += graph.nodes[src]['pixel count']
    graph.nodes[dst]['mean color'] = (graph.nodes[dst]['total color'] /
                                      graph.nodes[dst]['pixel count'])

def run_rag_merging(labels, image, thresh=0.2, connectivity=2):
    g = graph.rag_mean_color(image, labels, connectivity=connectivity)
    img = image.copy()
    img[np.isnan(img)] = -1 # fill 
    labels2 = graph.merge_hierarchical(labels, g, thresh=thresh, rag_copy=False,
                                       in_place_merge=True,
                                       merge_func=merge_mean_color,
                                       weight_func=_weight_mean_color)
    return labels2

### Make a scheme to store/organize these segmentation results

def process_watershed_results(result_dict):
    temp = pd.DataFrame()
    # push from "image" format to (y,x) coordinate unfolded DF ready format
    for key, val in result_dict.items():
        temp[key] = pd.DataFrame(val).stack()

    temp.index = temp.index.set_names(['y','x'])
    temp = temp.reorder_levels([1,0])
    
    return temp

def run_watershed_and_store(
    image_dict,
    image_subset_id='nonstroma',
    smoothing_mode='median',
    denoise_disk_size=4,
    grad_disk_size=1,
    gradient_cutoff=45,
    thresh=0.2,
    use_markers=True,
    **kwargs
):
    water_out = run_watershed_dict(image_dict, image_subset_id=image_subset_id, smoothing_mode=smoothing_mode, 
                                   denoise_disk_size=denoise_disk_size, grad_disk_size=grad_disk_size, 
                                   gradient_cutoff=gradient_cutoff, use_markers=use_markers)
    labels = water_out['labels']
    gradient = water_out['gradient']
    merged_labels = run_rag_merging(labels, image_dict[image_subset_id], thresh=thresh)
    water_out['merged_labels'] = merged_labels

        
    return water_out
    


def merge_nonoverlapping(df1, df2):
    cols_to_use = df2.columns.difference(df1.columns)
    return df1.join(df2[cols_to_use])

def compare_segments(df):
    """
    df: has columns for segment size ('count'), and mean prob. value ('mean')
    
    """    
    overall_mean = df['mean'].mean()
    df['rel_mean'] = (df['mean'] - overall_mean)/overall_mean

    top_mean = df.loc[df['count'].idxmax(),'mean']
    df['rel_top'] = (df['mean'] - top_mean)/top_mean
    
    return df

def get_top_k_segments(df, k=2, sort_col='count'):
    return df.sort_values(sort_col,ascending=False).iloc[:k,:]

def get_roi_tile_stitching(df, x_start, y_start, window_size, tile_size = 512, interpolate=False, tumor_cutoff=0.5, path_var='full_path', 
                           overlay_var='smoothed_prob_g4_not_g2'):
    map_size = window_size+1
    image_map = np.zeros((map_size*tile_size,map_size*tile_size, 3))

    subset = df.loc[(df.x >= x_start) & (df.x <= x_start+window_size) & (df.y >= y_start) & (df.y <= y_start+window_size)]
    if tumor_cutoff is not None:
        subset = subset.loc[subset['smoothed_prob_tumor'] > tumor_cutoff]

    subset['y_rel'] = subset['y'] - y_start
    subset['x_rel'] = subset['x'] - x_start

    dataset = SlideDataset(paths=subset[path_var].values, 
                           slide_ids=subset[['x_rel','y_rel']].values, 
                           labels=subset[['x','y']].values, 
                           transform_compose=transforms.ToTensor())

    print('grabbing tiles...')
    for img, (x_abs, y_abs), (x_rel,y_rel) in dataset:
        adjust_x = x_rel*tile_size
        adjust_y = y_rel*tile_size
        image_map[adjust_y:adjust_y+tile_size, adjust_x:adjust_x+tile_size] = img.permute(1,2,0).numpy()

    roi = image_map

    x = torch.sparse_coo_tensor(subset[['y_rel','x_rel']].values.transpose(), subset[overlay_var].values).to_dense()
    if interpolate:
        print(f'using smoothed predictions and interpolated heatmap ({overlay_var})')
        x_resize = resize(x, roi.shape[:2], order=1)
    else:
        print(f'using smoothed predictions and raw heatmap ({overlay_var})')
        x_resize = resize(x, roi.shape[:2], order=0)
        
    return roi, x, x_resize

def check_example(plotting_df, margin=5, window_size=5, overlay_var=['smoothed_prob_g4_not_g2'], path_var='full_path', tumor_cutoff=None, vmin=0, vmax=1, invert_yaxis=False):
    """
    only grabs H&E tiles 
    """
    
    set_rc(5,5, font_scale=1.8)
    plot_assignment_clusters_combined(plotting_df, use_style=False, hue_order=HUE_ORDER, marker_size=15, invert_yaxis=invert_yaxis)
    plt.show()
    
    keep_plotting = True
    while keep_plotting:
        x_start, y_start = [int(x) for x in input('x_start y_start').split()]


        region_checking = True
        while region_checking:
            set_rc(12,5, font_scale=1.8)
            fig, axes = plt.subplots(1,2)
            plot_assignment_clusters_combined(plotting_df, use_style=False, hue_order=HUE_ORDER, ax=axes[0], marker_size=15, invert_yaxis=invert_yaxis)
            plotting_df_zoom = plotting_df.loc[(plotting_df.x>(x_start-margin)) & (plotting_df.x<(x_start+window_size+margin)) & (plotting_df.y>(y_start-margin)) & (plotting_df.y<(y_start+window_size+margin))]
            plot_assignment_clusters_combined(plotting_df_zoom, use_style=False, hue_order=HUE_ORDER, ax=axes[1], marker_size=50, invert_yaxis=invert_yaxis)
            for ax in axes:
                ax.add_patch(Rectangle((x_start, y_start), window_size, window_size, color='gray', alpha=0.5))
            plt.show()
        
            if input('OK Region?') == 'y':
                region_checking = False
            else:
                x_start, y_start = [int(x) for x in input('re-enter x_start y_start').split()]   
                plotting_df_zoom = plotting_df.loc[(plotting_df.x>(x_start-margin)) & (plotting_df.x<(x_start+window_size+margin)) & (plotting_df.y>(y_start-margin)) & (plotting_df.y<(y_start+window_size+margin))]
                plot_assignment_clusters_combined(plotting_df_zoom, use_style=False, hue_order=HUE_ORDER, ax=axes[1], marker_size=50, invert_yaxis=invert_yaxis)
                for ax in axes:
                    ax.add_patch(Rectangle((x_start, y_start), window_size, window_size, color='gray', alpha=0.5))
                plt.show()
                
        set_rc(30,len(overlay_var)*7)
        fig,axes = plt.subplots(len(overlay_var),4, squeeze=False)
        for row_idx, overlay in enumerate(overlay_var):
            he_roi, pred_map, pred_map_resize = get_roi_tile_stitching(plotting_df, x_start=x_start, y_start=y_start,window_size=window_size, tumor_cutoff=tumor_cutoff, path_var=path_var, overlay_var=overlay)
            CMAP='BrBG'
            sns.heatmap(pred_map, ax=axes[row_idx, 0], annot=True, square=True, cbar=False, vmin=vmin, vmax=vmax)
            axes[row_idx,1].imshow(pred_map_resize, vmin=vmin, vmax=vmax, cmap=CMAP)
            axes[row_idx,2].imshow(he_roi)
            axes[row_idx,2].imshow(pred_map_resize, vmin=vmin, vmax=vmax, cmap=CMAP, alpha=0.25)
            axes[row_idx,3].imshow(he_roi)
        if not invert_yaxis:
            for ax in axes.ravel():
                ax.invert_yaxis()
        plt.show()
        
        if input('Continue?') == 'y':
            keep_plotting = True
        else:
            keep_plotting = False

def quad_plot_rag(plotting_df, labels, nonstroma_image, full_image, gradient, rag, mark_boundaries=False, plotting_mode='overlay',edge_width=2.5):
    fig, axes = plt.subplots(2,2, sharex=True, sharey=True)

    plot_assignment_clusters_combined(plotting_df, use_style=False, marker_size=25, ax=axes[0,1], invert_yaxis=True)
    axes[0,1].set_title('Initial Bin Labels')

    out = color.label2rgb(labels, nonstroma_image, kind=plotting_mode, bg_label=0)
    if mark_boundaries:
        out = segmentation.mark_boundaries(out, labels, (0, 0, 0))
    axes[0,0].imshow(out, alpha=0.6)  
    axes[0,0].set_title('Segment Labels')
    
    axes[1,0].imshow(gradient, cmap=plt.cm.nipy_spectral, alpha=0.6)
    axes[1,0].set_title('Local Gradient')

    lc = graph.show_rag(labels, rag, nonstroma_image, border_color='black', 
                        edge_width=edge_width,
                        img_cmap='gray', edge_cmap='viridis', ax=axes[1,1])
    axes[1,1].imshow(images['stroma'] != -1, alpha=0.2)
    axes[1,1].set_title('RAG Connections')
    cbar = plt.colorbar(lc, fraction=0.03, ax=axes[1,1], boundaries=np.linspace(0,0.25,25))
    
    return fig

# def rag_plot(plotting_df, labels, nonstroma_image, full_image, gradient, rag, mark_boundaries=False, plotting_mode='overlay',edge_width=2.5):
#     fig, ax = plt.subplots(1,1, sharex=True, sharey=True)


#     lc = graph.show_rag(labels, rag, nonstroma_image, border_color='black', 
#                         edge_width=edge_width,
#                         img_cmap='gray', edge_cmap='viridis', ax=ax)
#     ax.imshow(images['stroma'] != -1, alpha=0.2)
#     ax.set_title('RAG Connections')
#     cbar = plt.colorbar(lc, fraction=0.03, ax=ax, boundaries=np.linspace(0,0.25,25))
    
#     return fig

def rag_plot(plotting_df, labels, nonstroma_image, full_image, gradient, rag, ax, mark_boundaries=False, plotting_mode='overlay',edge_width=2.5, min_segment_size=None, hue_order=None, 
             bg_color=50, img_cmap='gray'):
    out = color.label2rgb(labels, nonstroma_image, kind=plotting_mode, bg_label=0, bg_color=(bg_color,bg_color,bg_color), image_alpha=1)

    # remove small segments if specified
    if min_segment_size is not None:
        g = rag.copy()
        to_remove = []
        for node_name in g.nodes:
            if g.nodes[node_name]['pixel count'] < min_segment_size:
                to_remove.append(node_name)
        for node_name in to_remove:
            g.remove_node(node_name)
    else:
        g = rag
    
    lc = graph.show_rag(labels, g, full_image, border_color='black', 
                        edge_width=edge_width,
                        img_cmap=img_cmap, edge_cmap='viridis', ax=ax)
    ax.set_title('RAG Connections')
    cbar = plt.colorbar(lc, fraction=0.03, ax=ax, boundaries=np.linspace(0,1.,25))

    ax.imshow(out, alpha=0.6)  
    flipped_centroids = {node_key: (y,x) for node_key, (x,y) in nx.get_node_attributes(g,'centroid').items()}  
    nx.draw_networkx(g, pos=flipped_centroids, ax=ax)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    
#     return fig


def invert_rag_weight(weight):
    return np.sqrt(np.power(weight, 2)/3)



def watershed_pipeline_dask(unique_id=None, df=None, thresh=None, **kwargs):
    images = get_separated_heatmaps(df.reset_index())
    out = run_watershed_and_store(image_dict=images, thresh=thresh, **kwargs)

    calls = process_watershed_results(out)
    df = df.join(calls)
    df['merge_thresh'] = thresh
    if np.all(df['labels'] == df['merged_labels']):
        df['merged_labels'] = np.nan
        
    df['unique_id'] = unique_id
    out['unique_id'] = unique_id
    images['unique_id'] = unique_id
    
    results = {'seg_out':out, 'seg_df':df, 'seg_images':images, 'thresh':thresh, 'unique_id':unique_id}
    return results

def summarize_nucleus_calls(df, unique_id):
    df_tilemean = df.groupby(['tx','ty']).mean()
    df_tilemean['stroma_tumor_ratio'] = df.groupby(['tx','ty']).apply(lambda x: get_stroma_tumor_ratio(x))
    df_tilemean['stroma_til_ratio'] = df.groupby(['tx','ty']).apply(lambda x: get_til_tumor_ratio(x))

    df_tilemean['tumor_counts'] = df.groupby(['tx','ty']).apply(lambda x: get_cell_counts(x,'tumor'))
    df_tilemean['til_counts'] = df.groupby(['tx','ty']).apply(lambda x: get_cell_counts(x,'til'))
    df_tilemean['stroma_counts'] = df.groupby(['tx','ty']).apply(lambda x: get_cell_counts(x,'stroma'))
    df_tilemean['unique_id'] = unique_id
    return df_tilemean

def summarize_nucleus_calls_cohort_level(df, patient_var='unique_id'):
    df_tilemean = df.groupby([patient_var,'tx','ty']).mean()
    df_tilemean['stroma_tumor_ratio'] = df.groupby([patient_var, 'tx','ty']).apply(lambda x: get_stroma_tumor_ratio(x))
    df_tilemean['stroma_til_ratio'] = df.groupby([patient_var, 'tx','ty']).apply(lambda x: get_til_tumor_ratio(x))

    df_tilemean['tumor_counts'] = df.groupby([patient_var, 'tx','ty']).apply(lambda x: get_cell_counts(x,'tumor'))
    df_tilemean['til_counts'] = df.groupby([patient_var, 'tx','ty']).apply(lambda x: get_cell_counts(x,'til'))
    df_tilemean['stroma_counts'] = df.groupby([patient_var, 'tx','ty']).apply(lambda x: get_cell_counts(x,'stroma'))

    return df_tilemean

import networkx as nx

def quad_plot_rag_node(plotting_df, labels, nonstroma_image, full_image, gradient, rag, mark_boundaries=False, plotting_mode='overlay',edge_width=2.5, min_segment_size=None, hue_order=None):
    fig, axes = plt.subplots(2,2, sharex=True, sharey=True)

    plot_assignment_clusters_combined(plotting_df, use_style=False, marker_size=25, ax=axes[0,1], invert_yaxis=True, hue_order=hue_order)
    axes[0,1].set_title('Initial Bin Labels')

    out = color.label2rgb(labels, nonstroma_image, kind=plotting_mode, bg_label=0)
    if mark_boundaries:
        out = segmentation.mark_boundaries(out, labels, (0, 0, 0))
    axes[0,0].imshow(out, alpha=0.6)  
    axes[0,0].set_title('Segment Labels')
    
    axes[1,0].imshow(gradient, cmap=plt.cm.nipy_spectral, alpha=0.6)
    axes[1,0].set_title('Local Gradient')
    
    # remove small segments if specified
    if min_segment_size is not None:
        g = rag.copy()
        to_remove = []
        for node_name in g.nodes:
            if g.nodes[node_name]['pixel count'] < min_segment_size:
                to_remove.append(node_name)
        for node_name in to_remove:
            g.remove_node(node_name)
    else:
        g = rag
    
    lc = graph.show_rag(labels, g, full_image, border_color='black', 
                        edge_width=edge_width,
                        img_cmap='gray', edge_cmap='viridis', ax=axes[1,1])
    axes[1,1].set_title('RAG Connections')
    cbar = plt.colorbar(lc, fraction=0.03, ax=axes[1,1], boundaries=np.linspace(0,0.25,25))

    axes[1,1].imshow(out, alpha=0.6)  
    flipped_centroids = {node_key: (y,x) for node_key, (x,y) in nx.get_node_attributes(g,'centroid').items()}  
    nx.draw_networkx(g, pos=flipped_centroids, ax=axes[1,1])
    axes[1,1].tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

    
def post_process_seg_graph(outputs, graph_name='rag', min_segment_size=10, node_diff_cutoff=0.25):
    unique_id = outputs['unique_id']
    repacked_outs = {
        'plotting_df':outputs['seg_df'], 'labels':outputs['seg_out']['merged_labels'], 'nonstroma_image':outputs['seg_images']['nonstroma'], 'full_image':outputs['seg_images']['nonstroma'], 
        'gradient':outputs['filled_grad'], 'rag':outputs[graph_name]
    }

    g = outputs['rag'].copy()

    to_remove = []
    for node_name in g.nodes:
        if g.nodes[node_name]['pixel count'] < min_segment_size:
            to_remove.append(node_name)
    for node_name in to_remove:
        g.remove_node(node_name)
        
    # get area of considered tumor segments
    tumor_tile_count = np.sum([v['pixel count'] for k,v  in g.nodes.items()])
    

        
    for node_id, node in g.nodes.items():
        node_adj = g.adj[node_id]
        node_mean = node['mean color'][0]
        
        # annotate relative fraction that a node represents of total tumor tile area
        node_rep_frac = node['pixel count'] / tumor_tile_count
        node['tumor_area_frac'] = node_rep_frac

        if (len(node_adj) == 0) & (node['pixel count'] > min_segment_size):
            node['connection_category'] = 'isolated'
            
        if len(node_adj) == 1:
            node['connection_category'] = 'single_connection'

        elif len(node_adj) > 1:
            node['connection_category'] = 'multi_connection'

        if 'connection_category' not in node.keys():
            node['connection_category'] = 'other'
            
            
    for k,v in g.edges.items():
        edge_diff = invert_rag_weight(v['weight'])
        v['diff'] = edge_diff
        if edge_diff > node_diff_cutoff:
            v['diff_category'] = 'major'
        else:
            v['diff_category'] = 'minor'
            
        v['tumor_area_frac_sum'] = g.nodes[k[0]]['tumor_area_frac'] + g.nodes[k[1]]['tumor_area_frac']
        
        v['node0_tumor_area_frac'] = g.nodes[k[0]]['tumor_area_frac']
        v['node1_tumor_area_frac'] = g.nodes[k[1]]['tumor_area_frac']

    return g

def assign_graph_summary(unique_id, g, summary_labels, seg_df, min_segment_size=10, node_diff_cutoff=0.25):
    edges = [x for x in g.edges.items()]
    nodes = [x for x in g.nodes.items()]

    # look for cases without edges / adjacency in RAG
    if all([node['connection_category'] == 'isolated' for node_id, node in nodes]):
        if len(nodes) == 1:
            summary_labels[unique_id] = 'single_foci'
        if len(nodes) > 1:
            summary_labels[unique_id] = 'multiple_disconnected_foci'
    
    # for cases with adjacency edges present, bin based on how large an edge weight
    detected_major = any([v['diff_category'] == 'major' for k,v in g.edges.items()])
    if detected_major:
        summary_labels[unique_id] = 'major_diff_candidate'
    if (len(edges) > 0) & (not detected_major):
        summary_labels[unique_id] = 'minor_diff_candidate'


def get_separated_heatmaps_simplified(plotting_df, col):
    x_max, y_max = plotting_df[['x','y']].max()
    placeholder = np.ones((y_max+1, x_max+1))*-1
    for (x,y), row in plotting_df.set_index(['x','y']).iterrows():
        placeholder[y,x] = row[col]

    return placeholder


def assign_til_status_label(count, lower_cutoff=5, upper_cutoff=25):
    if count < lower_cutoff:
        label = 'non_infiltrated'
    elif count > upper_cutoff:
        label = 'highly_infiltrated'
    else:
        label = 'intermed_infiltrated'
    return label

def assign_til_status_label_exp(count, seg_size, frac_cutoff=0.25, tile_area_cutoff=50):
    frac_infiltrated = count/seg_size
#     print(count, frac_infiltrated)
    if (count < tile_area_cutoff) & (frac_infiltrated < frac_cutoff):
        label = 'non_infiltrated'
    else:
        if frac_infiltrated < frac_cutoff:
            label = 'localized_infiltration'
        if frac_infiltrated >= frac_cutoff:
            label = 'dispersed_infiltration'
    return label
    

def cutoff_counter(x, cutoff):
    return (x > cutoff).sum()

# def add_TIL_info(g, til_annotations,  frac_cutoff=0.25, tile_area_cutoff=20):
#     """
#     assumes a RAG already processed by `post_process_seg_graph`
    
#     """
#     # add node level info for TILs
#     for nidx, node in g.nodes.items():
#         try:
#             g.nodes[nidx]['til_status_combined'] = til_annotations.loc[nidx, 'til_status_combined']
#         except Exception as e :
#             print(e)
#             print(f'skipped node {nidx}')


#     # check if any difference along existing adjacency edges in RAG 
#     for k,v in g.edges.items():
#         try:
#             n0, n1 = k
#             n0_status = g.nodes[n0]['til_status_combined']
#             n1_status = g.nodes[n1]['til_status_combined']
            
#             node0_grade = g.nodes[k[0]]['mean color'][0]
#             node1_grade = g.nodes[k[1]]['mean color'][0]
#             lower_grade_node_rel_idx = np.argmin([node0_grade, node1_grade])
#             lower_node_idx = k[lower_grade_node_rel_idx]
#             higher_grade_node_rel_idx = np.argmax([node0_grade, node1_grade])
#             higher_node_idx = k[higher_grade_node_rel_idx]
            
#             status_set = set([n0_status, n1_status])
#             ordered_labels = list(status_set)
#             ordered_labels = [x.split('_')[0] for x in ordered_labels]

#             if n0_status != n1_status:
# #                 print(f'contrasting til statuses {n0} {n0_status}, {n1} {n1_status}')
#                 if status_set == set(['localized_infiltration','non_infiltrated']):
#                     g.edges[k]['til_contrast_category'] = 'non_infiltrated_bordering_localized'
#                     if g.nodes[lower_node_idx]['til_status_combined'] == 'non_infiltrated':
#                         g.edges[k]['combined_edge_context'] = 'lower_grade_non_infiltrated_higher_grade_localized'
    
#                     else:
#                         g.edges[k]['combined_edge_context'] = 'higher_grade_non_infiltrated_lower_grade_localized'
                        
                        
#                 if ('non_infiltrated' in status_set) & ('localized_infiltration' not in status_set):
#                     g.edges[k]['til_contrast_category'] = 'non_infiltrated_bordering_dispersed'
#                     if g.nodes[lower_node_idx]['til_status_combined'] == 'non_infiltrated':
#                         g.edges[k]['combined_edge_context'] = 'lower_grade_non_infiltrated_higher_grade_dispersed'
#                     else:
#                         g.edges[k]['combined_edge_context'] = 'higher_grade_non_infiltrated_lower_grade_dispersed'                    
                    

#                 if ('non_infiltrated' not in status_set) & ('localized_infiltration' in status_set):
#                     g.edges[k]['til_contrast_category'] = 'localized_bordering_dispersed'
#                     if g.nodes[lower_node_idx]['til_status_combined'] == 'localized_infiltration':
#                         g.edges[k]['combined_edge_context'] = 'lower_grade_localized_higher_grade_dispersed'
#                     else:
#                         g.edges[k]['combined_edge_context'] = 'higher_grade_localized_lower_grade_dispersed'
                    
#                 if ('non_infiltrated' not in status_set) & ('localized_infiltration' not in status_set):
#                     g.edges[k]['til_contrast_category'] = f'contrasting_dispersed_{ordered_labels[0]}_bordering_{ordered_labels[1]}'
                    
#                     if  til_annotations.loc[lower_node_idx, 'tiles_above_til_cutoff'] > til_annotations.loc[higher_node_idx, 'tiles_above_til_cutoff']:
#                         g.edges[k]['combined_edge_context'] = 'lower_grade_more_infiltrated_both_dispersed'
#                     else:
#                         g.edges[k]['combined_edge_context'] = 'higher_grade_more_infiltrated_both_dispersed'
          
        
#                 if  til_annotations.loc[lower_node_idx, 'tiles_above_til_cutoff'] > til_annotations.loc[higher_node_idx, 'tiles_above_til_cutoff']:
#                     g.edges[k]['general_edge_context'] = 'lower_grade_more_infiltrated'
#                 else:
#                     g.edges[k]['general_edge_context'] = 'higher_grade_more_infiltrated'
                        
                        
#             if n0_status == n1_status:
#                 g.edges[k]['til_contrast_category'] = f'same_dispersed_{ordered_labels[0]}_bordering_{ordered_labels[1]}'
                
# #             print(g.edges[k]['combined_edge_context'])
# #             print(g.edges[k]['til_contrast_category'], '\n')
#         except Exception as e:
#             print(e)
#             print(f'skipped edge {k}')
            
#     return g


def check_example_mod(plotting_df, margin=5, window_size=5, overlay_var=['til_counts','smoothed_prob_g4_not_g2'], path_vars=['full_path'], tumor_cutoff=None, vmin=0, vmax=1, invert_yaxis=False):
    """
    can grab arbitrary image tiles 
    uses `get_separated_heatmaps_simplified` to make sns heatmaps rather than sns.scatterplots

    """
    subplot_store = {}
    n_cols = len(overlay_var)
    set_rc(n_cols*5, 5, font_scale=1.8)
    fig, axes = plt.subplots(1, n_cols)
    
    for idx, col in enumerate(overlay_var):
        heatmap = get_separated_heatmaps_simplified(plotting_df.reset_index().rename(columns={'tx':'x', 'ty':'y'}), col=col)
        if col.endswith('counts'):
            vmax=100
        elif col.endswith('ratio'):
            vmax=5
        else:
            vmax=None
        sns.heatmap(heatmap, vmax=vmax, ax=axes[idx])
        axes[idx].set_title(col)
        
        subplot_store[f'heatmap_{col}'] = heatmap
        
    plt.show()
    
    keep_plotting = True
    while keep_plotting:
        x_start, y_start = [int(x) for x in input('x_start y_start').split()]


        region_checking = True
        while region_checking:
            set_rc(12,5, font_scale=1.8)
            fig, axes = plt.subplots(1,2, sharex=False, sharey=False)
            sns.heatmap(heatmap, vmax=vmax, ax=axes[0])


            heatmap_zoom = heatmap[y_start:(y_start+window_size), x_start:(x_start+window_size)]
            sns.heatmap(heatmap_zoom, vmax=vmax, ax=axes[1])     
            for ax in axes:
                overlay_rect = Rectangle((x_start, y_start), window_size, window_size, color='gray', alpha=0.5)
                ax.add_patch(overlay_rect)
            plt.show()
        
            if input('OK Region?') == 'y':
                region_checking = False
            else:
                x_start, y_start = [int(x) for x in input('re-enter x_start y_start').split()]   
                heatmap_zoom = heatmap[y_start:(y_start+window_size), x_start:(x_start+window_size)]

                sns.heatmap(heatmap_zoom, vmax=vmax, ax=axes[1])

                for ax in axes:
                    overlay_rect = Rectangle((x_start, y_start), window_size, window_size, color='gray', alpha=0.5)
                    ax.add_patch(overlay_rect)
                plt.show()
        
        for path_var in path_vars:
            for row_idx, overlay in enumerate(overlay_var):
                roi, pred_map, pred_map_resize = get_roi_tile_stitching(plotting_df, x_start=x_start, y_start=y_start,window_size=window_size, tumor_cutoff=tumor_cutoff, path_var=path_var, overlay_var=overlay)
                
                subplot_store[f'roi_{path_var}'] = roi
                subplot_store[f'pred_map_{overlay}'] = pred_map

        
        set_rc(len(subplot_store)*5, 5, font_scale=1.5)
        fig, axes = plt.subplots(1,len(subplot_store))
        axes = axes.reshape(-1)
        for rel_idx, (k,v) in enumerate(subplot_store.items()):
            axes[rel_idx].imshow(v)
            axes[rel_idx].set_title(k)
        
        axes = axes.reshape(1,-1)
        plt.show()
        
        if input('Continue?') == 'y':
            keep_plotting = True
        else:
            keep_plotting = False
        
    overlay_rect_clone = Rectangle((x_start, y_start), window_size, window_size, color='green', alpha=0.5)
    return subplot_store, overlay_rect_clone

# 20210513NB
def summarize_nodes(g, seg_df):
    df = seg_df.groupby('merged_labels').mean()
    edges = [x for x in g.edges.items()]
    nodes = [x for x in g.nodes.items()]
    
    rag_edge_categories = [v['diff_category'] for k,v in g.edges.items()]
    rag_edge_categories = pd.Series(rag_edge_categories, name='rag_edge_categories')
    
    cc = [x for x in nx.connected_components(g)]
    degrees = pd.Series({k:v for k,v in g.degree}, name='degree')
    conn_cats = pd.Series({k:v['connection_category'] for k,v in g.nodes.items()}, name='connection_category')
    df = df.join(conn_cats).dropna(subset=['connection_category'])
    df = df.join(degrees)
    avg_degree = degrees.mean()
    return {'seg_df':df, 'cc':cc, 'rag_edge_categories':rag_edge_categories, 'avg_degree':avg_degree}

# 20210514NB 
def get_multi_connection_context(g):
    context_descriptions = {}
    for node_id, deg in g.degree:
        desc = 'unclassified'
        if deg > 1:
            node_mean = g.nodes[node_id]['mean color'][0]
            comparison_means = []
            for adj_node_id, adj_node_info in g.adj[node_id].items():
                if adj_node_info['diff_category'] == 'major':
                    comparison_means.append(g.nodes[adj_node_id]['mean color'][0])
            if all([node_mean > x for x in comparison_means]):
                desc = 'high_amidst_low'
            if all([node_mean < x for x in comparison_means]):
                desc = 'low_amidst_high'
            
        if deg == 1:
            node_mean = g.nodes[node_id]['mean color'][0]
            comparison_means = []
            for adj_node_id, adj_node_info in g.adj[node_id].items():
                if adj_node_info['diff_category'] == 'major':
                    comparison_means.append(g.nodes[adj_node_id]['mean color'][0])
            if all([node_mean > x for x in comparison_means]):
                desc = 'high_bordering_low'
            if all([node_mean < x for x in comparison_means]):
                desc = 'low_bordering_high'
            
        if deg == 0:
            desc = 'isolated'
            
        context_descriptions[node_id] = desc
        
    return pd.Series(context_descriptions, name='multi_connection_context')


def summarize_nucleus_calls_dask(df, unique_id, fpath):
    nuc_subset = filter_results(df)
    add_ws_coords(nuc_subset)
    nuc_subset['ws_x'] = nuc_subset['ws_x'] - nuc_subset['ws_x'].min()
    nuc_subset['ws_y'] = nuc_subset['ws_y'] - nuc_subset['ws_y'].min()
    nuc_subset['unique_id'] = unique_id
    
    summary = summarize_nucleus_calls(nuc_subset, unique_id)
    summary.to_csv(fpath)
    
#     return {'filtered_nuclei':nuc_subset, 'summary':summary}

def summarize_nucleus_calls_dask_manual_intensity(df, unique_id, fpath, intensity_cutoff=60.):
    nuc_subset = filter_results(df)
    add_ws_coords(nuc_subset)
    nuc_subset['ws_x'] = nuc_subset['ws_x'] - nuc_subset['ws_x'].min()
    nuc_subset['ws_y'] = nuc_subset['ws_y'] - nuc_subset['ws_y'].min()
    nuc_subset['unique_id'] = unique_id
    
    # override previous TIL overcalling with a more stringent intensity requirement
    nuc_subset.loc[(nuc_subset.tissue_class == 'til') & (nuc_subset.intensity > intensity_cutoff), 'tissue_class'] = 'tumor'
    
    summary = summarize_nucleus_calls(nuc_subset, unique_id)
    summary.to_csv(fpath)
    


def rotate_sns_labels(g, rotation=30):
    g.set_xticklabels(rotation=rotation)

def aggregate_counts(df, col, index_name='unique_id',):
    count_store = pd.DataFrame()
    for uid, df in df.groupby(index_name):
        y = df[col].value_counts()
        y.name = uid
        y = pd.DataFrame(y)
        count_store = pd.concat([count_store, y.transpose()])
    return count_store

def add_TIL_info(g, til_annotations,  frac_cutoff=0.25, tile_area_cutoff=20):
    """
    assumes a RAG already processed by `post_process_seg_graph`
    
    """
    # add node level info for TILs
    for nidx, node in g.nodes.items():
        try:
            g.nodes[nidx]['til_status_combined'] = til_annotations.loc[nidx, 'til_status_combined']
        except Exception as e :
            print(e)
            print(f'skipped node {nidx}')


    # check if any difference along existing adjacency edges in RAG 
    for k,v in g.edges.items():
        try:
            n0, n1 = k
            n0_status = g.nodes[n0]['til_status_combined']
            n1_status = g.nodes[n1]['til_status_combined']
            
            node0_grade = g.nodes[k[0]]['mean color'][0]
            node1_grade = g.nodes[k[1]]['mean color'][0]
            lower_grade_node_rel_idx = np.argmin([node0_grade, node1_grade])
            lower_node_idx = k[lower_grade_node_rel_idx]
            higher_grade_node_rel_idx = np.argmax([node0_grade, node1_grade])
            higher_node_idx = k[higher_grade_node_rel_idx]
            
            status_set = set([n0_status, n1_status])
            ordered_labels = list(status_set)
            ordered_labels = [x.split('_')[0] for x in ordered_labels]
            if len(ordered_labels) == 1: # catch cases where same label present for both nodes
                ordered_labels = ordered_labels+ordered_labels
                
            if n0_status != n1_status:
                if status_set == set(['localized_infiltration','non_infiltrated']):
                    g.edges[k]['til_contrast_category'] = 'non_infiltrated_bordering_localized'
                    if g.nodes[lower_node_idx]['til_status_combined'] == 'non_infiltrated':
                        g.edges[k]['combined_edge_context'] = 'lower_grade_non_infiltrated_higher_grade_localized'
    
                    else:
                        g.edges[k]['combined_edge_context'] = 'higher_grade_non_infiltrated_lower_grade_localized'
                        
                        
                if ('non_infiltrated' in status_set) & ('localized_infiltration' not in status_set):
                    g.edges[k]['til_contrast_category'] = 'non_infiltrated_bordering_dispersed'
                    if g.nodes[lower_node_idx]['til_status_combined'] == 'non_infiltrated':
                        g.edges[k]['combined_edge_context'] = 'lower_grade_non_infiltrated_higher_grade_dispersed'
                    else:
                        g.edges[k]['combined_edge_context'] = 'higher_grade_non_infiltrated_lower_grade_dispersed'                    
                    

                if ('non_infiltrated' not in status_set) & ('localized_infiltration' in status_set):
                    g.edges[k]['til_contrast_category'] = 'localized_bordering_dispersed'
                    if g.nodes[lower_node_idx]['til_status_combined'] == 'localized_infiltration':
                        g.edges[k]['combined_edge_context'] = 'lower_grade_localized_higher_grade_dispersed'
                    else:
                        g.edges[k]['combined_edge_context'] = 'higher_grade_localized_lower_grade_dispersed'
                    
                if ('non_infiltrated' not in status_set) & ('localized_infiltration' not in status_set):
                    g.edges[k]['til_contrast_category'] = f'contrasting_dispersed_{ordered_labels[0]}_bordering_{ordered_labels[1]}'
                    
                    if  til_annotations.loc[lower_node_idx, 'tiles_above_til_cutoff'] > til_annotations.loc[higher_node_idx, 'tiles_above_til_cutoff']:
                        g.edges[k]['combined_edge_context'] = 'lower_grade_more_infiltrated_both_dispersed'
                    else:
                        g.edges[k]['combined_edge_context'] = 'higher_grade_more_infiltrated_both_dispersed'
          
        
                if  til_annotations.loc[lower_node_idx, 'tiles_above_til_cutoff'] > til_annotations.loc[higher_node_idx, 'tiles_above_til_cutoff']:
                    g.edges[k]['general_edge_context'] = 'lower_grade_more_infiltrated'
                else:
                    g.edges[k]['general_edge_context'] = 'higher_grade_more_infiltrated'
                        
                        
            if n0_status == n1_status:
                g.edges[k]['til_contrast_category'] = 'same_infiltration_status'
                g.edges[k]['combined_edge_context'] = f'same_{ordered_labels[0]}_bordering_{ordered_labels[1]}'
                g.edges[k]['general_edge_context'] = 'same_infiltration_status'

        except Exception as e:
            print(e)
            print(f'skipped edge {k}')
            
    return g

def compare_nodes(df, slidemean_subset, source_col='smoothed_prob_g4_not_g2', out_col='grade'):
    """
    df: has columns for segment size ('count'), and mean prob. value ('mean')
    
    """    
    df[f'rel_{source_col}'] = (df[source_col] - slidemean_subset[source_col]) / slidemean_subset[source_col]

    df.loc[df[f'rel_{source_col}'] > 0.05, f'rel_{out_col}_label'] = f'{out_col}_above_mean'
    df.loc[df[f'rel_{source_col}'] < -0.05, f'rel_{out_col}_label'] = f'{out_col}_below_mean'
    df[f'rel_{out_col}_label'] = df[f'rel_{out_col}_label'].fillna(f'{out_col}_near_mean')
    
    return df[[f'rel_{source_col}',f'rel_{out_col}_label']]

def check_subplot_zoom(subplots, x_start, y_start, tile_size=512, fig_size=10, font_scale=1, heatmap_col_0='pred_map_til_counts', heatmap_col_1='pred_map_smoothed_prob_g4_not_g2'):
    img = subplots['roi_full_path']
    img = (img * 255).astype(np.uint8)

    til_heatmap = subplots['pred_map_til_counts']
    grade_heatmap = subplots['pred_map_smoothed_prob_g4_not_g2']

    x_start_px = int(x_start*tile_size)
    y_start_px = int(y_start*tile_size)
    zoom_size = tile_size
    x_zoom_size = int(2*zoom_size)

    img_zoom = img[y_start_px:y_start_px+zoom_size,x_start_px:x_start_px+x_zoom_size]


    set_rc(fig_size, fig_size, font_scale)
    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    wide_ax = fig.add_subplot(gs[1, :])

    
    sns.heatmap(til_heatmap, ax=ax0, vmax=50, annot=True, square=True, cbar=False)
    ax0.set_title('TIL Counts')
    ax1.imshow(img)
    ax1.set_title(f'Rel. (X,Y) = ({x_start}, {y_start})')
    sns.heatmap(grade_heatmap, ax=ax2, vmin=0, vmax=1, annot=True, square=True, cbar=False)

    ax2.set_title('Smoothed Grade Score')
    
    
    overlay_rect = Rectangle((x_start, y_start), 2, 1, color='gray', alpha=0.5)
    ax0.add_patch(overlay_rect)
    overlay_rect = Rectangle((x_start, y_start), 2, 1, color='yellow', alpha=0.15)
    ax2.add_patch(overlay_rect)

    wide_ax.imshow(img_zoom)
    return fig

def summarize_nucleus_calls_til_only(df, unique_id):
    df_tilemean = df.groupby(['tx','ty']).mean()
    df_tilemean['til_counts'] = df.groupby(['tx','ty']).apply(lambda x: get_cell_counts(x,'til'))
    df_tilemean['unique_id'] = unique_id
    return df_tilemean

def summarize_nucleus_calls_dask_manual_intensity_til_only(df, unique_id, fpath, intensity_cutoff=60.):
    nuc_subset = filter_results(df)
    add_ws_coords(nuc_subset)
    nuc_subset['ws_x'] = nuc_subset['ws_x'] - nuc_subset['ws_x'].min()
    nuc_subset['ws_y'] = nuc_subset['ws_y'] - nuc_subset['ws_y'].min()
    nuc_subset['unique_id'] = unique_id
    
    # override previous TIL overcalling with a more stringent intensity requirement
    nuc_subset.loc[(nuc_subset.tissue_class == 'til') & (nuc_subset.intensity > intensity_cutoff), 'tissue_class'] = 'tumor'
    
    summary = summarize_nucleus_calls_til_only(nuc_subset, unique_id)
    summary.to_csv(fpath)

def check_example_multi(plotting_df, margin=5, window_size=5, overlay_var=['til_counts','smoothed_prob_g4_not_g2'], path_vars=['full_path'], 
                        tumor_cutoff=None, vmin=0, vmax=1, invert_yaxis=False):
    """
    can grab arbitrary image tiles 
    uses `get_separated_heatmaps_simplified` to make sns heatmaps rather than sns.scatterplots
    
    stores multiple subplot groups if you specify to continue

    """
    subplot_store = {}
    n_cols = len(overlay_var)
    set_rc(n_cols*5, 5, font_scale=1.8)
    fig, axes = plt.subplots(1, n_cols)
    
    for idx, col in enumerate(overlay_var):
        heatmap = get_separated_heatmaps_simplified(plotting_df.reset_index().rename(columns={'tx':'x', 'ty':'y'}), col=col)
        if col.endswith('counts'):
            vmax=100
        elif col.endswith('ratio'):
            vmax=5
        else:
            vmax=None
        sns.heatmap(heatmap, vmax=vmax, ax=axes[idx])
        axes[idx].set_title(col)
        
        subplot_store[f'heatmap_{col}'] = heatmap
        
    plt.show()
    
    keep_plotting = True
    while keep_plotting:
        temp_roi_group = {}
        x_start, y_start = [int(x) for x in input('x_start y_start').split()]


        region_checking = True
        while region_checking:
            set_rc(12,5, font_scale=1.8)
            fig, axes = plt.subplots(1,2, sharex=False, sharey=False)
            sns.heatmap(heatmap, vmax=vmax, ax=axes[0])


            heatmap_zoom = heatmap[y_start:(y_start+window_size), x_start:(x_start+window_size)]
            sns.heatmap(heatmap_zoom, vmax=vmax, ax=axes[1])     
            for ax in axes:
                overlay_rect = Rectangle((x_start, y_start), window_size, window_size, color='gray', alpha=0.5)
                ax.add_patch(overlay_rect)
            plt.show()
        
            if input('OK Region?') == 'y':
                region_checking = False
            else:
                x_start, y_start = [int(x) for x in input('re-enter x_start y_start').split()]   
                heatmap_zoom = heatmap[y_start:(y_start+window_size), x_start:(x_start+window_size)]

                sns.heatmap(heatmap_zoom, vmax=vmax, ax=axes[1])

                for ax in axes:
                    overlay_rect = Rectangle((x_start, y_start), window_size, window_size, color='gray', alpha=0.5)
                    ax.add_patch(overlay_rect)
                plt.show()
        
        for path_var in path_vars:
            for row_idx, overlay in enumerate(overlay_var):
                roi, pred_map, pred_map_resize = get_roi_tile_stitching(plotting_df, x_start=x_start, y_start=y_start,window_size=window_size, tumor_cutoff=tumor_cutoff, path_var=path_var, overlay_var=overlay)
                
                temp_roi_group[f'roi_{path_var}'] = roi
                temp_roi_group[f'pred_map_{overlay}'] = pred_map

        
        set_rc(len(subplot_store)*5, 5, font_scale=1.5)
        fig, axes = plt.subplots(1,len(temp_roi_group))
        axes = axes.reshape(-1)
        for rel_idx, (k,v) in enumerate(temp_roi_group.items()):
            axes[rel_idx].imshow(v)
            axes[rel_idx].set_title(k)
        
        axes = axes.reshape(1,-1)
        plt.show()
        
        subplot_store[f'roi_{x_start}_{y_start}'] = temp_roi_group
        if input('Continue?') == 'y':
            keep_plotting = True
        else:
            keep_plotting = False
        
    overlay_rect_clone = Rectangle((x_start, y_start), window_size, window_size, color='green', alpha=0.5)
    return subplot_store, overlay_rect_clone

def get_qq(df, col1, col2, col_suffix='_quantiles', intervals=100):
    qq = pd.DataFrame()
    quantiles = np.linspace(0,1,intervals)
    qq['quantile'] = quantiles
    for col in [col1, col2]:
        temp_quantiles = [df[col].quantile(x) for x in quantiles]
        qq[f'{col}{col_suffix}'] = temp_quantiles
    return qq 

def describe_ccf(df):
    subclonal_drivers = (df.ccf_hat <= 0.5).sum()
    clonal_drivers = (df.ccf_hat > 0.5).sum()
    eps = 1
    score = subclonal_drivers/(clonal_drivers+eps)
    return score


def describe_ccf_mod(df):
    subclonal_drivers = ((df.ccf_hat <= 0.5) & (df.ccf_hat > 0.1)).sum()
    clonal_drivers = (df.ccf_hat > 0.5).sum()
    eps = 1
    score = subclonal_drivers/(clonal_drivers+eps)
    return score


def call_ith_braun(df):
    subclonal_drivers = (df.ccf_hat < 1.0).sum()
    clonal_drivers = (df.ccf_hat == 1.0).sum()
#     eps = 1
#     score = subclonal_drivers/(clonal_drivers+eps)
    score = subclonal_drivers/(clonal_drivers)
    return score


def weighted_median(df, val, weight):
    """
    https://stackoverflow.com/questions/26102867/python-weighted-median-algorithm-with-pandas
    """
    try:
        df_sorted = df.sort_values(val)
        cumsum = df_sorted[weight].cumsum()
        cutoff = df_sorted[weight].sum() / 2.
        weighted_median = df_sorted[cumsum >= cutoff][val].iloc[0].astype(int)
    except:
        weighted_median = np.nan
    return weighted_median


def calculate_wgii_revised(group_df):
    """
    df is a patient level subset of absolute annotated CN segment file (Ex. TCGA_mastercalls.abs_segtabs.fixed.txt for TCGA)
    """
    df = group_df.copy()
    try:
        df['chrom_rel_length'] = df.groupby('Chromosome').Length.apply(lambda x: x/x.sum())
        df['is_altered_cn'] = (df['Modal_Total_CN'] != df['wmedian_ploidy']).astype(float)
        df['is_altered_cn_length_weighted'] = df['is_altered_cn'] * df['chrom_rel_length']
        wgii = df.groupby('Chromosome').is_altered_cn_length_weighted.sum().mean()
        
    except:
        wgii = np.nan
        
    return wgii

from scipy.stats import entropy

def calculate_ccf_entropy(df, ccf_col='ccf_hat'):
    subclonal_drivers = ((df.ccf_hat <= 0.5) & (df.ccf_hat > 0.1)).sum()
    clonal_drivers = (df.ccf_hat > 0.5).sum()
    eps = 1
    score = subclonal_drivers/(clonal_drivers+eps)
    return score

from skimage.morphology import disk
import cv2
import random as rng


def get_segment_contours(df,tumor_col='smoothed_prob_tumor', tumor_cutoff=0.5):
    """
    Single main contour case
    Operates on heatmap of tile-level values and grabs the largest detectable contour (ie, will exclude small artifacts or subregions)
    """
    heatmap = get_separated_heatmaps_simplified(df.reset_index(), tumor_col)

    binary = ((heatmap > tumor_cutoff) * 255).astype(np.uint8)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ctr_lengths = [len(x) for x in contours]
    largest_ctr_idx = np.array(ctr_lengths).argmax()
    ctr_subset = contours[largest_ctr_idx]
        
    color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
    sub_im = np.zeros(heatmap.shape + (3,))
    cv2.fillPoly(sub_im, pts=[ctr_subset], color=color)
    flattened_subim = (sub_im > 0 ).sum(-1)
    
    return flattened_subim


def run_erosion_dilation(flattened_subim, kernel):
    erosion = skimage.morphology.erosion(flattened_subim, selem=kernel)
    dilation = skimage.morphology.dilation(flattened_subim, selem=kernel)

    y, x = np.where((dilation - erosion) > 0 )
    re = pd.DataFrame()
    re['y'] = y
    re['x'] = x
    re['tctm_label'] = 'margin'

    y, x = np.where(erosion > 0 )
    center_df = pd.DataFrame()
    center_df['y'] = y
    center_df['x'] = x
    center_df['tctm_label'] = 'center'

    updates = pd.concat([re, center_df]).set_index(['x','y'])
    return updates


def run_margin_calling(df, seg_label, kernel=disk(1), min_segment_size=50, tumor_col='smoothed_prob_tumor', tumor_cutoff=0.5):
    """
    Single main contour case
    """
    
    flattened_subim = get_segment_contours(df, tumor_col)
    if flattened_subim.sum() >= min_segment_size:
        out = run_erosion_dilation(flattened_subim, kernel)
        out['seg_label'] = seg_label
        return out
     

def run_margin_calling_dask(df, uid, outdir='./tctm_calls', kernel=disk(1), kernel_size=None, kernel_type=None, min_segment_size=50, tumor_col='smoothed_prob_tumor', tumor_cutoff=0.5):
    update_agg = []

    for seg_label, subdf in df.loc[df.merged_labels != 0].groupby('merged_labels'):
        updates = run_margin_calling(subdf, seg_label, kernel, min_segment_size)
        update_agg.append(updates)

    updates = run_margin_calling(df.loc[df.merged_labels != 0], 'all_tumor', kernel, min_segment_size)
    updates = updates.rename(columns={'tctm_label':'all_tumor_tctm_label'})
    update_agg.append(updates)

    update_agg = pd.concat(update_agg)
    update_agg['combined_seg_tctm'] = update_agg['seg_label'].astype(str) + '_' + update_agg['tctm_label'].fillna('nan')
    update_agg['agg_combined_seg_tctm'] = update_agg.loc[update_agg.seg_label !='all_tumor'].groupby(['x','y']).combined_seg_tctm.apply(lambda x: '_'.join(x))

    df = merge_nonoverlapping(df, update_agg)

    df['agg_combined_seg_tctm'] = df['agg_combined_seg_tctm'].fillna('stroma')
    df['all_tumor_tctm_label'] = df['all_tumor_tctm_label'].fillna('other')
    df['kernel_size'] = kernel_size
    df['kernel_type'] = kernel_type
    
    df.to_csv(f'{outdir}/{uid}_tctm_calls_nonTIL.csv')
    
#     return df

from itertools import combinations

def construct_km_multiplot(df, col_groups, stratifier_cols=['ICI','Non-ICI'], strat_col_name='drug_type', min_group_size=0, duration_var='os'):
    event_var = f'{duration_var}_event'
    n_subplots = len(col_groups)
    
    set_rc(int(10*n_subplots),7, font_scale=1.2)
    fig,axes = plt.subplots(1,2*n_subplots, sharex=True, sharey=True)
    
    for rel_idx, cols in enumerate(col_groups):
        for col_idx, strat in enumerate(stratifier_cols):
            temp_sub = df.loc[df[strat_col_name] == strat]

            for labels, subdf in temp_sub.groupby(cols):
                if len(subdf) >= min_group_size:
                    kmf = KaplanMeierFitter()
                    formatted_labels = [label.replace('_',' ').capitalize() for label in labels]
                    if type(labels) == list:
                        combined_label = ' + '.join(formatted_labels) + f'(n={len(subdf)})'
                    else:
                        combined_label = f'{labels} (n={len(subdf)})'

                    kmf.fit(durations=subdf[duration_var], event_observed=subdf[event_var], label=combined_label)
                    g = kmf.plot_survival_function(show_censors=True, ci_show=False, linewidth=3, ax=axes[col_idx+2*rel_idx]) 
                    move_leg(g,x=-3+(2*col_idx), y=-.5-col_idx)
                else:
                    print(f'skipping {strat} {labels} !')

                cols_break = " \n ".join(cols)
                axes[col_idx+2*rel_idx].set_title(f'CM-025 {strat.upper()} Arm \n {duration_var.upper()} \n {cols_break}')
                
    return fig

def construct_km_duo(df, col_groups, stratifier_cols=['ICI','Non-ICI'], strat_col_name='drug_type', min_group_size=0, duration_var='os',
                        leg_x_offset_scale=0.2, leg_y_offset_scale=0.75, leg_y_offset_bias=0.5):
    event_var = f'{duration_var}_event'
    n_subplots = len(col_groups)
    
    set_rc(int(12*n_subplots),7, font_scale=2)
    sns.set_style('white')
    fig,axes = plt.subplots(1,2*n_subplots, sharex=True, sharey=True)
    
    for rel_idx, cols in enumerate(col_groups):
        for col_idx, strat in enumerate(stratifier_cols):
            temp_sub = df.loc[df[strat_col_name] == strat]

            for labels, subdf in temp_sub.groupby(cols):
                if len(subdf) >= min_group_size:
                    kmf = KaplanMeierFitter()
                    formatted_labels = [label.replace('_',' ').capitalize() for label in labels]
                    if type(labels) == list:
                        combined_label = ' + '.join(formatted_labels) + f'(n={len(subdf)})'
                    else:
                        combined_label = f'{labels} (n={len(subdf)})'

                    kmf.fit(durations=subdf[duration_var], event_observed=subdf[event_var], label=combined_label)
                    g = kmf.plot_survival_function(show_censors=True, ci_show=False, linewidth=3, ax=axes[col_idx+2*rel_idx]) 
                    move_leg(g, x=(leg_x_offset_scale*col_idx), y=-leg_y_offset_bias-leg_y_offset_scale*col_idx)
                else:
                    print(f'skipping {strat} {labels} !')

                cols_break = " \n ".join(cols)
                axes[col_idx+2*rel_idx].set_title(f'CM-025 {strat.upper()} Arm \n {duration_var.upper()} \n {cols_break}')
    return fig

def compare_edge_DIE_calls(row, col_a='edge0_manual_infiltrated_call', col_b='edge1_manual_infiltrated_call'):
    if row[col_a] == row[col_b]:
        return f'both_{row[col_a]}'
    else:
        diff_set = '_'.join(list(set([row[col_a],row[col_b]])))
        return f'diff_contact_{diff_set}'

def calculate_group_ratio(df, num_group_cols=2, col='benefit', group_a='CB', group_b='NCB'):
    ratio = df.loc[group_a] / df.loc[group_b]
    return ratio

from scipy.stats import fisher_exact, binom_test

def calc_cb_ncb_fisher_ici_vs_nonici(df, formula, col_a='ICI_CB', col_b='ICI_NCB', col_c='Non-ICI_CB', col_d='Non-ICI_NCB', alternative='two-sided'):
    conting = np.array([
        df.loc[formula][[col_a,col_b]],
        df.loc[formula][[col_c,col_d]],
    ]).transpose()
    oddsr, p = fisher_exact(conting, alternative=alternative)
    return p


def calc_binom_p(df, formula, col_a='CB_ICI', col_b='NCB_ICI', alternative='two-sided'):
    return binom_test(df.loc[formula][[col_a,col_b]], alternative=alternative)

def calc_fisher_alt(contingency_table, alternative='two-sided'):
    oddsr, p = fisher_exact(contingency_table, alternative=alternative)
    return oddsr, p



def check_example_multi_mod(plotting_df, unique_id, outdir='./roi_outputs', margin=5, window_size=5, overlay_var=['til_counts','smoothed_prob_g4_not_g2'], path_vars=['full_path'], 
                        tumor_cutoff=None, vmin=0, vmax=1, invert_yaxis=False, save_figs=True, plot_x_scale=40, plot_height=40, font_scale=3.):
    """
    can grab arbitrary image tiles 
    uses `get_separated_heatmaps_simplified` to make sns heatmaps rather than sns.scatterplots
    
    stores multiple subplot groups if you specify to continue

    """
    subplot_store = {}
    n_cols = len(overlay_var)
    set_rc(n_cols*5, 5, font_scale=1.8)
    fig, axes = plt.subplots(1, n_cols)
    
    vmax_mapper = {'roi_full_path':vmax}
    
    for idx, col in enumerate(overlay_var):
        heatmap = get_separated_heatmaps_simplified(plotting_df.reset_index().rename(columns={'tx':'x', 'ty':'y'}), col=col)
        if col.endswith('counts'):
            vmax_mapper[col] = 100
        elif col.endswith('ratio'):
            vmax_mapper[col]=5
        elif col == 'gradient':
            vmax_mapper[col]=50
        else:
            vmax_mapper[col]=vmax
        sns.heatmap(heatmap, vmax=vmax_mapper[col], ax=axes[idx])
        axes[idx].set_title(col)
        
        subplot_store[f'heatmap_{col}'] = heatmap
        
    plt.show()
    
    keep_plotting = True
    while keep_plotting:
        temp_roi_group = {}
        x_start, y_start = [int(x) for x in input('x_start y_start').split()]


        region_checking = True
        while region_checking:
            set_rc(12,5, font_scale=font_scale)
            fig, axes = plt.subplots(1,2, sharex=False, sharey=False)
            sns.heatmap(heatmap, vmin=vmin, vmax=vmax, ax=axes[0])


            heatmap_zoom = heatmap[y_start:(y_start+window_size), x_start:(x_start+window_size)]
            sns.heatmap(heatmap_zoom, vmin=vmin, vmax=vmax, ax=axes[1])     
            for ax in axes:
                overlay_rect = Rectangle((x_start, y_start), window_size, window_size, color='gray', alpha=0.5)
                ax.add_patch(overlay_rect)
            if save_figs:
                plt.savefig(f'{outdir}/{unique_id}_global_heatmap_{x_start}_{y_start}_overlay.png')
            plt.show()
            if input('OK Region?') == 'y':
                region_checking = False
                
            else:
                x_start, y_start = [int(x) for x in input('re-enter x_start y_start').split()]   
                heatmap_zoom = heatmap[y_start:(y_start+window_size), x_start:(x_start+window_size)]

                sns.heatmap(heatmap_zoom, vmax=vmax, ax=axes[1])

                for ax in axes:
                    overlay_rect = Rectangle((x_start, y_start), window_size, window_size, color='gray', alpha=0.5)
                    ax.add_patch(overlay_rect)
                plt.show()
        
        for path_var in path_vars:
            for row_idx, overlay in enumerate(overlay_var):
                roi, pred_map, pred_map_resize = get_roi_tile_stitching(plotting_df, x_start=x_start, y_start=y_start,window_size=window_size, tumor_cutoff=tumor_cutoff, path_var=path_var, overlay_var=overlay)
                
                temp_roi_group[f'roi_{path_var}'] = roi
                temp_roi_group[f'pred_map_{overlay}'] = pred_map

        
        set_rc(len(subplot_store)*plot_x_scale, plot_height, font_scale=font_scale)
        fig, axes = plt.subplots(1,len(temp_roi_group))
        axes = axes.reshape(-1)
        for rel_idx, (col,v) in enumerate(temp_roi_group.items()):
            if col == 'roi_full_path':
                axes[rel_idx].imshow(v)
                if save_figs:
                    solo_roi_image = Image.fromarray((v*255).astype(np.uint8))
                    solo_roi_image.save(f'{outdir}/{unique_id}_roi_windowsize{window_size}_margin{margin}_{x_start}_{y_start}_solo_hne_roi.png')
            else:
                print(col, vmax_mapper[col.split('pred_map_')[-1]])
                axes[rel_idx].imshow(v, vmin=vmin, vmax=vmax_mapper[col.split('pred_map_')[-1]])
#                 sns.heatmap(v, ax=axes[rel_idx], vmin=vmin, vmax=vmax_mapper[col.split('pred_map_')[-1]])
                
            axes[rel_idx].set_title(col)
        
        axes = axes.reshape(1,-1)
#         plt.tight_layout()
        if save_figs:
            plt.savefig(f'{outdir}/{unique_id}_roi_windowsize{window_size}_margin{margin}_{x_start}_{y_start}.png')
        plt.show()
        
        subplot_store[f'roi_{x_start}_{y_start}'] = temp_roi_group
        if input('Continue?') == 'y':
            keep_plotting = True
        else:
            keep_plotting = False
        
    overlay_rect_clone = Rectangle((x_start, y_start), window_size, window_size, color='green', alpha=0.5)
        
    return subplot_store, overlay_rect_clone


def get_qbins_simple(df, col, num_bins):
    bin_series, bins = pd.qcut(df[col], retbins=True, q=np.linspace(0,1,num_bins+1), labels=False,
                              duplicates='drop')
    return bin_series, bins


def quad_plot_rag_node_mod(plotting_df, labels, nonstroma_image, full_image, gradient, rag, mark_boundaries=False, plotting_mode='overlay',
                           edge_width=2.5, min_segment_size=None, min_edge_diff=None, image_alpha=0.8):
    fig, axes = plt.subplots(2,2, sharex=True, sharey=True)

    plot_assignment_clusters_combined(plotting_df, use_style=False, marker_size=25, ax=axes[0,1], invert_yaxis=True)
    axes[0,1].set_title('Initial Bin Labels')

    out = color.label2rgb(labels, nonstroma_image, kind=plotting_mode, bg_label=0)
    if mark_boundaries:
        out = segmentation.mark_boundaries(out, labels, (0, 0, 0))
    axes[0,0].imshow(out, alpha=0.6)  
    axes[0,0].set_title('Segment Labels')
    
    axes[1,0].imshow(gradient, cmap=plt.cm.nipy_spectral, alpha=0.6)
    axes[1,0].set_title('Local Gradient')
    
    g = rag.copy()
    
    # remove small segments if specified
    if min_segment_size is not None:
        
        to_remove = []
        for node_name in g.nodes:
            if g.nodes[node_name]['pixel count'] < min_segment_size:
                to_remove.append(node_name)
        for node_name in to_remove:
            g.remove_node(node_name)
    
    # remove edges below a certain score difference
    if min_edge_diff is not None:
        to_remove = []
        for (node0, node1), edge in g.edges.items():
            if edge['diff'] < min_edge_diff:
                to_remove.append((node0, node1))
        for (node0, node1) in to_remove:
            g.remove_edge(node0, node1)
    
    lc = graph.show_rag(labels, g, full_image, border_color='black', 
                        edge_width=edge_width,
                        img_cmap='gray', edge_cmap='viridis', ax=axes[1,1])
    axes[1,1].set_title('RAG Connections')
    cbar = plt.colorbar(lc, fraction=0.03, ax=axes[1,1], boundaries=np.linspace(0,0.25,25))

    axes[1,1].imshow(out, alpha=0.6)  
    flipped_centroids = {node_key: (y,x) for node_key, (x,y) in nx.get_node_attributes(g,'centroid').items()}  
    nx.draw_networkx(g, pos=flipped_centroids, ax=axes[1,1])
    axes[1,1].tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    
    
def get_ecdf(series, interval_step):
    intervals = np.arange(0,1+interval_step, interval_step)
    ecdf = [(series <= x).mean() for x in intervals]
    
    plt.plot(intervals,ecdf)
    
    return intervals, ecdf


# 20210511NB 
def watershed_pipeline_extended_dask(unique_id=None, df=None, thresh=None, min_segment_size=1, connectivity_degree=2, **kwargs):
    """
    Also creates RAG and cleans up for downstream analysis
    """
    images = get_separated_heatmaps(df.reset_index())
    out = run_watershed_and_store(image_dict=images, thresh=thresh, **kwargs)

    calls = process_watershed_results(out)
    df = df.join(calls)
    df['merge_thresh'] = thresh
    if np.all(df['labels'] == df['merged_labels']):
        df['merged_labels'] = np.nan
        
    df['unique_id'] = unique_id
    out['unique_id'] = unique_id
    images['unique_id'] = unique_id

    
    images = get_separated_heatmaps(df.reset_index(), col='smoothed_prob_g4_not_g2')
    labels = get_separated_heatmaps(df.reset_index(), col='merged_labels')
    gradient = get_separated_heatmaps(df.reset_index(), col='gradient')
    
    ### run filling to allow RAG construction
    filled_labels = np.ma.masked_array(labels['all'], mask = labels['all'] == -1).filled(1000)  # fill in background values with label "1000"
    filled_labels = np.ma.masked_array(filled_labels, mask = np.isnan(filled_labels)).filled(2000).astype(int) # fill in stroma values with label "2000"

    filled_image = np.ma.masked_array(images['all'], mask = images['all'] == -1).filled(0.5)
    filled_image = np.ma.masked_array(filled_image, mask = np.isnan(filled_image)).filled(0.5)

    filled_grad= np.ma.masked_array(gradient['nonstroma'], mask = gradient['nonstroma'] == -1).filled(0)
    filled_grad = np.ma.masked_array(filled_grad, mask = np.isnan(filled_grad)).filled(0).astype(int)


    g = graph.rag_mean_color(image=filled_image, labels=filled_labels, connectivity=connectivity_degree)
    for placeholder_node in [0, 1000, 2000]:
        try:
            g.remove_node(placeholder_node)
        except:
            pass
#     g.remove_node(0) # remove background and stroma
#     g.remove_node(1000) # remove background and stroma
#     g.remove_node(2000) # remove background and stroma

    # remove small nodes 
    to_remove = []
    for node_name in g.nodes:
        if g.nodes[node_name]['pixel count'] < min_segment_size:
            to_remove.append(node_name)
    for node_name in to_remove:
        g.remove_node(node_name)

    # this does inversion multiple times so needs to be reworked differently
#     # reformat edges of RAG to represent absolute p() difference rather than pseudo euclidean 
#     g_clone = g.copy()
#     for idx, node_adj in g_clone.adj.items():
#         for sub_idx, sub_node_adj in node_adj.items():
#             sub_node_adj['weight'] = invert_rag_weight(sub_node_adj['weight'])

    results = {'seg_out':out, 'seg_df':df, 'seg_images':images, 'thresh':thresh, 'unique_id':unique_id,'rag':g, 'filled_grad':filled_grad, 'filled_image':filled_image}
    return results


def prune_nodes(rag, min_segment_size):
    if min_segment_size is not None:
        g = rag.copy()
        to_remove = []
        for node_name in g.nodes:
            if g.nodes[node_name]['pixel count'] < min_segment_size:
                to_remove.append(node_name)
        for node_name in to_remove:
            g.remove_node(node_name)
    else:
        g = rag
    return g




def watershed_pipeline_label_expansion(unique_id=None, df=None, thresh=None, min_segment_size=1, connectivity_degree=2, expansion_distance=1, **kwargs):
    """
    Also creates RAG and cleans up for downstream analysis
    """
    images = get_separated_heatmaps(df.reset_index())
    out = run_watershed_and_store(image_dict=images, thresh=thresh, **kwargs)

    calls = process_watershed_results(out)
    df = df.join(calls)
    df['merge_thresh'] = thresh
    if np.all(df['labels'] == df['merged_labels']):
        df['merged_labels'] = np.nan
        
    df['unique_id'] = unique_id
    out['unique_id'] = unique_id
    images['unique_id'] = unique_id

    images = get_separated_heatmaps(df.reset_index(), col='smoothed_prob_g4_not_g2')
    labels = get_separated_heatmaps(df.reset_index(), col='merged_labels')
    gradient = get_separated_heatmaps(df.reset_index(), col='gradient')
    
    ### run filling to allow RAG construction
    filled_labels = np.ma.masked_array(labels['all'], mask = labels['all'] == -1).filled(1000)  # fill in background values with label "1000"
    filled_labels = np.ma.masked_array(filled_labels, mask = np.isnan(filled_labels)).filled(2000).astype(int) # fill in stroma values with label "2000"

    filled_image = np.ma.masked_array(images['all'], mask = images['all'] == -1).filled(0.5)
    filled_image = np.ma.masked_array(filled_image, mask = np.isnan(filled_image)).filled(0.5)

    filled_grad= np.ma.masked_array(gradient['nonstroma'], mask = gradient['nonstroma'] == -1).filled(0)
    filled_grad = np.ma.masked_array(filled_grad, mask = np.isnan(filled_grad)).filled(0).astype(int)

    # create RAG based on segment-mean grade score
    g = graph.rag_mean_color(image=filled_image, labels=filled_labels, connectivity=connectivity_degree)
    
    # disconnect placeholder nodes from graph 
    for placeholder_node in [0, 1000, 2000]:
        try:
            g.remove_node(placeholder_node)
        except:
            pass

    # remove small nodes 
    to_remove = []
    for node_name in g.nodes:
        if g.nodes[node_name]['pixel count'] < min_segment_size:
            to_remove.append(node_name)
    for node_name in to_remove:
        g.remove_node(node_name)

    # create pseudo input and plot induced connections at a given expansion distance
    candidate_tumor_mask = get_raw_separated_heatmaps(df.reset_index(), cols=['putative_tumor'])['putative_tumor']
    candidate_tumor_mask[candidate_tumor_mask==-1] = candidate_tumor_mask.max()+1
    
    # recode segmentation labels such that stroma labeled regions are "background" to allow expansion
    mod_merged_labels = np.copy(out['merged_labels'])
    mod_merged_labels[candidate_tumor_mask==0] = 0
    mod_merged_labels[candidate_tumor_mask==(candidate_tumor_mask.max()+1)] = candidate_tumor_mask.max()+1
        
    # expand labels by `expansion_distance` pixels
    expanded_labels = expand_labels(mod_merged_labels, expansion_distance)
    pseudo_rag_input = np.copy(expanded_labels).astype(float)
    
    # create pseudo image in which the expanded label region is assigned the input segment's mean grade score 
    for node in g.nodes:
        temp_mean = g.nodes[node]['mean color'][0]
        pseudo_rag_input[expanded_labels == node] = temp_mean
    
    # constuct RAG based on expanded labels
    g2 = graph.rag_mean_color(image=pseudo_rag_input, labels=expanded_labels, connectivity=connectivity_degree)
    g2.remove_node(0)
    
    # collapse expanded RAG if possible 
    mod_pseudo_rag = np.copy(pseudo_rag_input)
    mod_pseudo_rag[mod_merged_labels == 0] = -1  # need to replace background with neg num to avoid creating a new label connected to background
    temp_g = graph.rag_mean_color(image=mod_pseudo_rag, labels=expanded_labels, connectivity=connectivity_degree)

    secondary_merge = graph.merge_hierarchical(expanded_labels, temp_g, thresh=thresh, rag_copy=False,
                                           in_place_merge=False,
                                           merge_func=merge_mean_color,
                                           weight_func=_weight_mean_color)
    
    # create final RAG using post-expansion + post-merging labels 
    g3 = graph.rag_mean_color(image=mod_pseudo_rag, labels=secondary_merge, connectivity=connectivity_degree)
    g3.remove_node(0)
    
    results = {'seg_out':out, 'seg_df':df, 'seg_images':images, 'thresh':thresh, 'unique_id':unique_id,'rag':g, 'filled_grad':filled_grad, 'filled_image':filled_image,
              'expanded_rag':g2, 'merged_expanded_rag':g3, 'merged_expanded_labels':secondary_merge}
    return results

def watershed_pipeline_label_expansion_multi(unique_id=None, df=None, thresh=None, min_segment_size=1, connectivity_degree=2, expansion_distances=[1,], img_col='smoothed_prob_g4_not_g2', **kwargs):
    """
    Also creates RAG and cleans up for downstream analysis
    Does multiple expansions and aggregates outputs for comparison downstream
    """
    uid = unique_id
    images = get_separated_heatmaps(df.reset_index())
    out = run_watershed_and_store(image_dict=images, thresh=thresh, **kwargs)

    calls = process_watershed_results(out)
    df = df.join(calls)
    df['merge_thresh'] = thresh
    if np.all(df['labels'] == df['merged_labels']):
        df['merged_labels'] = np.nan
        
    df['unique_id'] = unique_id
    out['unique_id'] = unique_id
    images['unique_id'] = unique_id

    images = get_separated_heatmaps(df.reset_index(), col=img_col)
    labels = get_separated_heatmaps(df.reset_index(), col='merged_labels')
    gradient = get_separated_heatmaps(df.reset_index(), col='gradient')
    
    ### run filling to allow RAG construction
    filled_labels = np.ma.masked_array(labels['all'], mask = labels['all'] == -1).filled(1000)  # fill in background values with label "1000"
    filled_labels = np.ma.masked_array(filled_labels, mask = np.isnan(filled_labels)).filled(2000).astype(int) # fill in stroma values with label "2000"

    filled_image = np.ma.masked_array(images['all'], mask = images['all'] == -1).filled(0.5)
    filled_image = np.ma.masked_array(filled_image, mask = np.isnan(filled_image)).filled(0.5)

    filled_grad= np.ma.masked_array(gradient['nonstroma'], mask = gradient['nonstroma'] == -1).filled(0)
    filled_grad = np.ma.masked_array(filled_grad, mask = np.isnan(filled_grad)).filled(0).astype(int)

    # create RAG based on segment-mean grade score
    g = graph.rag_mean_color(image=filled_image, labels=filled_labels, connectivity=connectivity_degree)
    
    # disconnect placeholder nodes from graph 
    for placeholder_node in [0, 1000, 2000]:
        try:
            g.remove_node(placeholder_node)
        except:
            pass

    # remove small nodes 
    to_remove = []
    for node_name in g.nodes:
        if g.nodes[node_name]['pixel count'] < min_segment_size:
            to_remove.append(node_name)
    for node_name in to_remove:
        g.remove_node(node_name)

    # create pseudo input and plot induced connections at a given expansion distance
    candidate_tumor_mask = get_raw_separated_heatmaps(df.reset_index(), cols=['putative_tumor'])['putative_tumor']
    candidate_tumor_mask[candidate_tumor_mask==-1] = candidate_tumor_mask.max()+1
    
    # recode segmentation labels such that stroma labeled regions are "background" to allow expansion
    mod_merged_labels = np.copy(out['merged_labels'])
    mod_merged_labels[candidate_tumor_mask==0] = 0
    mod_merged_labels[candidate_tumor_mask==(candidate_tumor_mask.max()+1)] = candidate_tumor_mask.max()+1
    
    premerge_expansion_graph_agg = {}
    premerge_expansion_label_agg = {}
    expansion_graph_agg = {}
    expansion_label_agg = {}
    # expand labels by `expansion_distance` pixels
    for dist in expansion_distances:
        try:
            expanded_labels = expand_labels(mod_merged_labels, dist)
            pseudo_rag_input = np.copy(expanded_labels).astype(float)

            # create pseudo image in which the expanded label region is assigned the input segment's mean grade score 
            for node in g.nodes:
                temp_mean = g.nodes[node]['mean color'][0]
                pseudo_rag_input[expanded_labels == node] = temp_mean

            # constuct RAG based on expanded labels
            g2 = rag_mean_color(image=pseudo_rag_input, labels=expanded_labels, connectivity=connectivity_degree)
            g2.remove_node(0)

            # collapse expanded RAG if possible 
            mod_pseudo_rag = np.copy(pseudo_rag_input)
            mod_pseudo_rag[mod_merged_labels == 0] = -1  # need to replace background with neg num to avoid creating a new label connected to background
            temp_g = rag_mean_color(image=mod_pseudo_rag, labels=expanded_labels, connectivity=connectivity_degree)

            secondary_merge = graph.merge_hierarchical(expanded_labels, temp_g, thresh=thresh, rag_copy=False,
                                                   in_place_merge=True,
                                                   merge_func=merge_mean_color,
                                                   weight_func=_weight_mean_color)

            # create final RAG using post-expansion + post-merging labels 
            g3 = rag_mean_color(image=mod_pseudo_rag, labels=secondary_merge, connectivity=connectivity_degree)
            g3.remove_node(0)

            premerge_expansion_graph_agg[dist] = g2
            premerge_expansion_label_agg[dist] = expanded_labels
            expansion_graph_agg[dist] = g3
            expansion_label_agg[dist] = secondary_merge
        except Exception as e:
            print(f'{uid} expansion @ {dist} failed')
            print(f'Error: {e}')
    
    df['meta'] = df['putative_tumor'].map({True:'tumor',False:'stroma'})

    results = {'seg_out':out, 'seg_df':df, 'seg_images':images, 'thresh':thresh, 'unique_id':unique_id,'rag':g, 'filled_grad':filled_grad, 'filled_image':filled_image,
              'merged_expanded_rags':expansion_graph_agg, 'merged_expanded_labels':expansion_label_agg, 
               'premerge_expansion_rags':premerge_expansion_graph_agg,'premerge_expansion_labels':premerge_expansion_label_agg}
    return results

def post_process_seg_graph_simplified(rag, unique_id, min_segment_size=10, node_diff_cutoff=0.25):
    """
    Differs from `post_process_seg_graph` by only requiring a RAG input and not all inputs of a segmentation pipeline output
    """
    g = rag.copy()

    to_remove = []
    for node_name in g.nodes:
        if g.nodes[node_name]['pixel count'] < min_segment_size:
            to_remove.append(node_name)
    for node_name in to_remove:
        g.remove_node(node_name)
        
    # get area of considered tumor segments
    tumor_tile_count = np.sum([v['pixel count'] for k,v  in g.nodes.items()])
    

        
    for node_id, node in g.nodes.items():
        node_adj = g.adj[node_id]
        node_mean = node['mean color'][0]
        
        # annotate relative fraction that a node represents of total tumor tile area
        node_rep_frac = node['pixel count'] / tumor_tile_count
        node['tumor_area_frac'] = node_rep_frac

        if (len(node_adj) == 0) & (node['pixel count'] > min_segment_size):
            node['connection_category'] = 'isolated'
            
        if len(node_adj) == 1:
            node['connection_category'] = 'single_connection'

        elif len(node_adj) > 1:
            node['connection_category'] = 'multi_connection'

        if 'connection_category' not in node.keys():
            node['connection_category'] = 'other'
            
            
    for k,v in g.edges.items():
        edge_diff = invert_rag_weight(v['weight'])
        v['diff'] = edge_diff
        if edge_diff > node_diff_cutoff:
            v['diff_category'] = 'major'
        else:
            v['diff_category'] = 'minor'
            
        v['tumor_area_frac_sum'] = g.nodes[k[0]]['tumor_area_frac'] + g.nodes[k[1]]['tumor_area_frac']
        
        v['node0_tumor_area_frac'] = g.nodes[k[0]]['tumor_area_frac']
        v['node1_tumor_area_frac'] = g.nodes[k[1]]['tumor_area_frac']

    return g

def check_label_set(x, reference):
    assert type(x) == set
    return np.all([x in reference for x in x])

def check_label_set_df(seg_df, edge_df, min_segment_size=50, label_col='merged_labels'):
    reference = get_indices(seg_df[label_col].value_counts() >= min_segment_size)
    edge_df['passing_edge'] = edge_df.edge_set.apply(lambda x: check_label_set(x, reference))
    return edge_df

def classify_distal_vs_proximal_edge(row, proximal_cols=[0,1], distal_cols=[10,25], diff_cutoff=0.2):
    """
    Asks whether an edge at a given expansion distance "passes" in terms of its grade score difference and minimum tile count in each segment 
    Expects NaNs if edge does not exist 
    Expects DF entries to be `diff` between node scores (invert_rag_weight(weight) for given edge)
    """
        
    if not np.all(row[proximal_cols].isna()):
        if max(row[proximal_cols]) >= diff_cutoff:
            return 'proximal'

    elif not np.all(row[distal_cols].isna()):
        if max(row[distal_cols]) >= diff_cutoff:
            return 'distal'
    else:
        return 'other'


def get_raw_separated_heatmaps(df, cols=['smoothed_prob_g4_not_g2'], **kwargs):
    x_max, y_max = df[['x','y']].max()
    
    placeholders = {}
    for col in cols:
        placeholder = np.ones((y_max+1, x_max+1))*-1
        for (x,y), row in df.set_index(['x','y']).iterrows():
            placeholder[y,x] = row[col]

        placeholders[col] = placeholder
    
    return placeholders

def watershed_pipeline_dask_mod(unique_id=None, df=None, thresh=None, **kwargs):
    images = get_raw_separated_heatmaps(df=df.reset_index(), **kwargs)
    out = run_watershed_and_store(image_dict=images, thresh=thresh, **kwargs)

    calls = process_watershed_results(out)
    df = df.join(calls)
    df['merge_thresh'] = thresh
    if np.all(df['labels'] == df['merged_labels']):
        df['merged_labels'] = np.nan
        
    df['unique_id'] = unique_id
    out['unique_id'] = unique_id
    images['unique_id'] = unique_id
    
    results = {'seg_out':out, 'seg_df':df, 'seg_images':images, 'thresh':thresh, 'unique_id':unique_id}
    return results

def run_two_stage_watershed_segmentation(input_df, uid, tumor_thresh=0.35, grade_thresh=0.35, min_tumor_seg_mean=0.7, min_tumor_segment_size=25, 
                                         connectivity_degree=1, exp_dists = [1,10,25], tumor_col='smoothed_prob_tumor', grade_col='smoothed_prob_g4_not_g2'):
        
        tumor_seg = watershed_pipeline_dask_mod(unique_id=uid, df=input_df, gradient_cutoff=None, use_markers=False, thresh=tumor_thresh, 
                                             image_subset_id=tumor_col, cols=[tumor_col, grade_col])

        # filter out likely stromal regions
        df = tumor_seg['seg_df'].rename(columns={'merged_labels':'tumor_seg_label'})
        df_tumor_segmean = df.groupby(['tumor_seg_label']).aggregate(['count','mean'])[tumor_col].sort_values('mean')
        df_tumor_segmean = df_tumor_segmean.loc[df_tumor_segmean['count'] >= min_tumor_segment_size]
        putative_tumor_segs = get_indices(df_tumor_segmean['mean'] >= min_tumor_seg_mean)
        df['putative_tumor'] = df['tumor_seg_label'].apply(lambda x: np.isin(x, putative_tumor_segs))
        df['meta'] = df['putative_tumor'].map({True:'nonstroma',False:'stroma'}) # follow "meta" convention to allow compatibility with heatmap image generation
        
        try:
            grade_out = watershed_pipeline_label_expansion_multi(unique_id=uid, df=df.drop(columns=['labels', 'gradient', 'markers']), 
                                                            gradient_cutoff=None, use_markers=False, thresh=grade_thresh, 
                                                             connectivity_degree = connectivity_degree, expansion_distances=exp_dists,
                                                            image_subset_id='nonstroma', img_col=grade_col)
            return grade_out
        
        except Exception as e:
            # if we fail it's probably at the grade segmentation stage, but still useful to return tumor segmentations
            partial_out = {}
            partial_out['seg_df']= df
            partial_out['error'] = e
            return partial_out

        
        

def scatterplot_flexible(df, x_var='x', y_var='y', hue_var='meta', size=6, use_style=False, hue_order=None, ax=None, marker_size=5, 
                                      legend='auto', legend_out=True, invert_yaxis=False):
    """
    Assumes that we will rescale the x,y distance to be [0,1]
    """
    g = sns.scatterplot(x_var,y_var, data=df, hue=hue_var, hue_order=hue_order, ax=ax, s=marker_size, legend=legend)

    if legend_out & (legend != False):
        g.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)
    if invert_yaxis:
        g.invert_yaxis()
        

        
def construct_km_single(df, cols, min_group_size=0, duration_var='os', ax=None, leg_loc='upper right'):
    event_var = duration_var+'_event'
    for labels, subdf in df.groupby(cols):
        if len(subdf) >= min_group_size:
            kmf = KaplanMeierFitter()
            formatted_labels = [label.replace('_',' ').title() for label in labels]
            if type(labels) == list:
                combined_label = ' + '.join(formatted_labels) + f'(n={len(subdf)})'
            else:
                combined_label = f'{labels} (n={len(subdf)})'

            kmf.fit(durations=subdf[duration_var], event_observed=subdf[event_var], label=combined_label)
            g = kmf.plot_survival_function(show_censors=True, ci_show=False, linewidth=3, ax=ax) 
#             move_leg(g,x=-3+(2*col_idx), y=-.5-col_idx)
            g.legend(loc=leg_loc)
    
    
    
def prepare_generic_subset(df, cohort, duration_var=None, mandatory_vars=[],
                         drop_g1=True, drop_met_site=True, keep_stage_iv=True, til_only_cases=False,
                         use_risk_group_info=True, gene=None, drop_intermed_group=False,
                         create_binary_outcome_label=True, relabel_edge_cats=True,
                         qbins=3,
                        ):
    print(f'selecting {cohort}')
    temp = df.loc[df.cohort.isin([cohort])]
    print(temp.shape)
    
    print('dropping unusable data (grade seg based)')
    temp = temp.loc[temp['usable_nontil_data']]
    print(temp.shape)
    
    if duration_var is not None:
        event_var = duration_var + '_event'
    
        print(f'dropping entries missing any in {[duration_var,event_var]}')
        temp = temp.dropna(subset=[duration_var,event_var])

    if len(mandatory_vars) > 0:
        print(f'dropping entries missing any in {mandatory_vars}')
        temp = temp.dropna(subset=mandatory_vars)

        
    if 'updated_nonstroma_grade_mean' in temp.columns:
        print('renaming updated_nonstroma_grade_mean to nonstroma_grade_mean temporarily')
        temp = temp.rename(columns={'updated_nonstroma_grade_mean':'nonstroma_grade_mean', 'nonstroma_grade_mean':'old_nonstroma_grade_mean'})
        print(temp.shape)
        
    if qbins is not None:
        grade_bins, bins = get_qbins(temp, 'nonstroma_grade_mean', qbins)
        print('grade quantile bins: ', bins)
        temp['quantile_bin'] = grade_bins

    print(temp.shape)
    print('dropping low tumor tile count cases')
    temp = temp.loc[temp.candidate_category !='other']
    print(temp.shape)
    if drop_g1:
        print('dropping G1 [keeping unannotated cases]')
        temp = temp.loc[temp.grade !='G1']
        print(temp.shape)
    if drop_met_site:
        print('dropping metastatic biopsies')
        temp = temp.loc[temp.primary_site]
        print(temp.shape)

    if not keep_stage_iv:
        temp = temp.loc[temp.stage != 'Stage IV']
        print(temp.shape)
    else:
        print('keeping all stages ')
        print(temp.shape)

    if gene != None:
        print(f'taking only {gene} Mut/WT info available cases')
        temp = temp.loc[temp[gene] != 'no_info']
        print(temp.shape)

    if relabel_edge_cats:
        print('converting bool edge presence labels to string')
        temp['any_rag_edge'] = temp['any_rag_edge'].map({'present':'low/hi foci contact', 'not_present':'no low/hi foci contact'})
        temp['any_rag_edge'] = temp['any_rag_edge'].str.title()

        temp['any_diff_edge'] = temp['any_diff_edge'].map({True:'low/hi presence', False:'no mixed foci presence'})
        temp['any_proximal_edge'] = temp['any_proximal_edge'].map({True:'low/hi foci contact [proximal]', False:'no low/hi foci contact [proximal]'})
        temp['any_distal_edge'] = temp['any_distal_edge'].map({True:'low/hi foci contact [distal]', False:'no low/hi foci contact [distal]'})
        temp['any_diff_edge'] = temp['any_diff_edge'].str.title()
        temp['any_proximal_edge'] = temp['any_proximal_edge'].str.title()
        temp['any_diff_edge'] = temp['any_diff_edge'].str.title()
    
    return temp

def prepare_cm025_subset(df, duration_var='os', drug_type='ICI', mandatory_vars=['age_at_diagnosis','gender'],
                         drop_g1=True, drop_met_site=True, keep_stage_iv=True, til_only_cases=False,
                         risk_type='MSKCC', use_risk_group_info=True, gene=None, drop_intermed_group=False,
                         outcome_type='benefit', create_binary_outcome_label=True, relabel_edge_cats=True,
                         qbins=3, anno=None, high_grade_filter=False,
                        ):
    event_var = duration_var + '_event'

    assert drug_type in {'ICI','Non-ICI','any'}
    
    print('selecting cm025 only')
    temp = df.loc[df.cohort.isin(['cm025'])]
    print(temp.shape)

    print('dropping unusable data (grade seg based)')
    temp = temp.loc[temp['usable_nontil_data']]
    print(temp.shape)
    
    if high_grade_filter:
        print('selecting only cases with 1+ high grade foci')
        temp = temp.loc[temp['high_grade_passing'] == 'high_grade_present']
        print(temp.shape)
    
    print(f'dropping entries missing any in {[duration_var,event_var]+mandatory_vars}')
    temp = temp.dropna(subset=[duration_var,event_var]+mandatory_vars)

        
    if 'updated_nonstroma_grade_mean' in temp.columns:
        print('renaming updated_nonstroma_grade_mean to nonstroma_grade_mean temporarily')
        temp = temp.rename(columns={'updated_nonstroma_grade_mean':'nonstroma_grade_mean', 'nonstroma_grade_mean':'old_nonstroma_grade_mean'})
        print(temp.shape)
        
    if qbins is not None:
        grade_bins, bins = get_qbins(temp, 'nonstroma_grade_mean', qbins)
        print('grade quantile bins: ', bins)
        temp['quantile_bin'] = grade_bins

    print(temp.shape)
    print('dropping low tumor tile count cases')
    temp = temp.loc[temp.candidate_category !='other']
    print(temp.shape)
    if drop_g1:
        print('dropping G1 [keeping unannotated cases]')
        temp = temp.loc[temp.grade !='G1']
        print(temp.shape)
    if drop_met_site:
        print('dropping metastatic biopsies')
        temp = temp.loc[temp.primary_site]
        print(temp.shape)

    if not keep_stage_iv:
        print('dropping Stage IV cases')
        temp = temp.loc[temp.stage != 'Stage IV']
        print(temp.shape)
    else:
        print('keeping all stages ')
        print(temp.shape)

    if til_only_cases:
        print('only keeping cases with TIL calls')
        temp = temp.loc[temp['usable_til_data']]
        print(temp.shape)

    if use_risk_group_info:
        print(f'only keeping cases {risk_type} risk group annotation')
        temp = temp.join(anno[risk_type]).dropna(subset=[risk_type])
        print(temp.shape)

    if gene != None:
        print(f'taking only {gene} Mut/WT info available cases')
        temp = temp.loc[temp[gene] != 'no_info']
        print(temp.shape)

    if drug_type != 'any':
        print(f'only taking {drug_type} arm')
        temp = temp.loc[temp.drug_type == drug_type]
        print(temp.shape)

    if drop_intermed_group:
        if outcome_type == 'recist':
            print('dropping SD category')
            temp = temp.loc[temp['recist'].isin(['CRPR','PD'])]
            print(temp.shape)

        else:
            print('dropping ICB category')
            temp = temp.loc[temp['benefit'].isin(['CB','NCB'])]
            print(temp.shape)

    if create_binary_outcome_label:
        temp['recist_is_crpr_not_pd'] = temp['recist'] == 'CRPR'
        temp['benefit_is_cb_not_ncb'] = temp['benefit'] == 'CB'
            
    if relabel_edge_cats:
        print('converting bool edge presence labels to string')
        temp['any_rag_edge'] = temp['any_rag_edge'].map({'present':'low/hi foci contact', 'not_present':'no low/hi foci contact'})
        temp['any_rag_edge'] = temp['any_rag_edge'].str.title()

        temp['any_diff_edge'] = temp['any_diff_edge'].map({True:'low/hi presence', False:'no mixed foci presence'})
        temp['any_proximal_edge'] = temp['any_proximal_edge'].map({True:'low/hi foci contact [proximal]', False:'no low/hi foci contact [proximal]'})
        temp['any_distal_edge'] = temp['any_distal_edge'].map({True:'low/hi foci contact [distal]', False:'no low/hi foci contact [distal]'})
        temp['any_diff_edge'] = temp['any_diff_edge'].str.title()
        temp['any_proximal_edge'] = temp['any_proximal_edge'].str.title()
        temp['any_diff_edge'] = temp['any_diff_edge'].str.title()
    
    return temp


def run_cph_feature_scaling(df, unscaled_cols):
    print('Running StandardScaler')
    scaler = StandardScaler()
    return scaler.fit_transform(df[unscaled_cols])


def glob_cols(df, keyword):
    return list(filter(lambda x: keyword in x, df.columns))


def join_covariates(covariates, delim=' + '):
    return delim.join(covariates)

def run_cph_fitting_dask(df, duration_var, event_var, covariate_formula, split_name, penalizer=0.1, l1_ratio=0.25, identifiers={}):
    out = run_cph_comparison_multivar(df, duration_var, event_var, covariate_formula=covariate_formula, penalizer=penalizer, l1_ratio=l1_ratio)
    
    return {'split_name':split_name, 'formula':covariate_formula,'cph':out, **identifiers} 



def prepare_distal_rag(raw_seg_outs, uid, edge_annotation_df, rag_type='premerge_expansion_rags', distal_exp_dist_max=25, min_segment_size=10, node_diff_cutoff=0.2):

    count_store = []
    weighted_info_store= []
    merged_descriptions = []
    node_summaries = {}
    processed_graphs = {}
    
    outs = raw_seg_outs
    rags = deepcopy(outs[rag_type])
    rags[0] = outs['rag']
    
    largest_expansion_dist_passing = max(list(rags.keys()))
    if largest_expansion_dist_passing < distal_exp_dist_max:
        print(f'could not find expansion dist @ {distal_exp_dist_max}, replacing with {largest_expansion_dist_passing}')
        distal_exp_dist_max = largest_expansion_dist_passing


    for dist, entry in rags.items():
        g = entry.copy()
        g = post_process_seg_graph_simplified(g, uid, min_segment_size, node_diff_cutoff)

        if dist == 0: # store info derived from base graph only; we only need the others for edge related info
            processed_graphs[uid] = g
            node_summaries[uid] = summarize_nodes(g, outs['seg_df'])
        if dist == 1:
            g_proximal = g.copy()

    g_distal = rags[distal_exp_dist_max].copy()

    for edge_key, val in rags[distal_exp_dist_max].edges.items():
        failing=True
        for passing_edge in [set(x) for x in edge_annotation_df.loc[edge_annotation_df.edge_class == 'distal'].index.values]:
            if set(edge_key) == passing_edge:
                print(f'found match: {edge_key}')
                failing=False
        if failing:
            g_distal.remove_edge(*edge_key)

    for edge_key, val in rags[distal_exp_dist_max].edges.items():
        failing=True
        for passing_edge in [set(x) for x in edge_annotation_df.loc[edge_annotation_df.edge_class == 'proximal'].index.values]:
            if set(edge_key) == passing_edge:
                try:
                    g_distal.remove_edge(*edge_key)
                    print('removed ', edge_key)
                except:
                    pass
                
    # prune distal rag to make sure we don't include artificially expanded tiny segs
    if min_segment_size is not None:
        g = rags[0].copy()
        to_remove = []
        for node_name in g.nodes:
            if g.nodes[node_name]['pixel count'] < min_segment_size:
                to_remove.append(node_name)
        for processed_graph in [g_proximal, g_distal]:
            for node_name in to_remove:
                try:
                    processed_graph.remove_node(node_name)
                except:
                    pass
                
    return {'g_proximal':g_proximal, 'g_distal':g_distal}

def plot_prox_vs_distal_rags(raw_seg_out, proximal_rag, distal_rag, cmap0='Reds', cmap1='Blues', min_segment_size=10, node_diff_cutoff=0.2):
    ex_out = raw_seg_out
    mod_im = np.ma.masked_array(ex_out['seg_images']['all'], mask = np.isnan(ex_out['seg_images']['all'])).filled(0)
    mod_labels = np.ma.masked_array(ex_out['seg_out']['merged_labels'], mask = np.isnan(ex_out['seg_images']['all'])).filled(ex_out['seg_out']['merged_labels'].max()+1)
    repacked_outs = {
        'plotting_df':ex_out['seg_df'], 'labels':mod_labels, 'nonstroma_image':mod_im, 'full_image':ex_out['seg_images']['nonstroma'], 
        'gradient':ex_out['filled_grad'], 'rag':proximal_rag
    }
    fig, axes = plt.subplots(1,2, figsize=(20,30))

    plot = rag_plot(**repacked_outs, ax=axes[0], img_cmap=cmap0, mark_boundaries=True, edge_width=5, min_segment_size=min_segment_size, hue_order=['stroma','tumor'], )

    repacked_outs = {
        'plotting_df':ex_out['seg_df'], 'labels':mod_labels, 'nonstroma_image':mod_im, 'full_image':ex_out['seg_images']['nonstroma'], 
        'gradient':ex_out['filled_grad'], 'rag':distal_rag
    }
    plot2 = rag_plot(**repacked_outs, ax=axes[1], img_cmap=cmap1, mark_boundaries=True, edge_width=5, min_segment_size=min_segment_size, hue_order=['stroma','tumor'], )

    axes[0].set_title('Proximal Connections')
    axes[1].set_title('Distal Connections')
    plt.tight_layout()
    return fig


def discretize_edge_scores(x):
    if np.isnan(x):
        label = 'no_edge'
    else:
        if x == 0:
            label = 'zero_score'
        if x > 0:
            label = 'pos_score'
        if x < 0:
            label = 'neg_score'
    return label

def discretize_edge_scores_alt(x):
    """
    Neg vs Non-neg instead of allowing zero group
    """
    if np.isnan(x):
        label = 'no_edge'
    else:
        if x >= 0:
            label = 'nonneg_score'
        if x < 0:
            label = 'neg_score'
    return label