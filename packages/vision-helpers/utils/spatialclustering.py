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

from skimage.filters.rank import entropy as local_entropy
from skimage.morphology import disk, square

# old
def run_vis(slide_id, df):
    temp_df = df.loc[slide_id]
    temp_df['x'] = temp_df['x'].astype(int)
    temp_df['y'] = temp_df['y'].astype(int)
    temp_grade_store = np.zeros((temp_df.y.max()+1, temp_df.x.max()+1)) - 1
    temp_tumor_store = np.zeros((temp_df.y.max()+1, temp_df.x.max()+1)) - 1
    temp_filtered_tumor_store = np.zeros((temp_df.y.max()+1, temp_df.x.max()+1)) - 1

    for idx, row in temp_df.iterrows():
        temp_grade_store[row.y, row.x] = row['g2_vs_g4']
        temp_tumor_store[row.y, row.x] = row['prob_tumor']
        if row['prob_tumor'] >= 0.7:
            temp_filtered_tumor_store[row.y, row.x] = row['g2_vs_g4']

    fig, axes = plt.subplots(1,3)
    sns.heatmap(temp_grade_store, vmin=-1, vmax=1, center=0, ax=axes[0], cbar=False)
    sns.heatmap(temp_tumor_store, vmin=-1, vmax=1, center=0, ax=axes[1], cbar=False)
    sns.heatmap(temp_filtered_tumor_store, vmin=-1, vmax=1, center=0, ax=axes[2], cbar=False)
    return fig


from sklearn.cluster import AgglomerativeClustering


def create_and_smoothen_heatmaps(slide_id, df, smoothing_fn, grade_score_factor=1, tumor_score_factor=1):
    maps = get_heatmap_inputs(slide_id, df)
    stack = torch.stack([torch.Tensor(x) for x in [maps['grade'], maps['tumor_raw']]])
    smoothed_stack = smoothing_fn(stack.unsqueeze(0))[0].numpy()
    
    temp = pd.concat([pd.DataFrame(smoothed_stack[0]).unstack(), pd.DataFrame(smoothed_stack[1]).unstack()], 1).reset_index()
    temp.columns = ['x','y','g2_vs_g4','prob_tumor']
    temp['g2_vs_g4_score'] = temp['g2_vs_g4']*grade_score_factor
    temp['tumor_score'] = temp['prob_tumor']*tumor_score_factor
    
    nonblank_temp = temp.loc[temp.g2_vs_g4 >= 0]
    nonblank_temp['x_scaled'] = nonblank_temp['x'] / nonblank_temp['x'].max()
    nonblank_temp['y_scaled'] = nonblank_temp['y'] / nonblank_temp['y'].max()
    
    return nonblank_temp, smoothed_stack, maps


def subsample_and_agg_cluster(data, num_subsamples=400, linkage='single', dist_threshold=0.05, n_clusters=None):
    subsamples = data.sample(num_subsamples)
    clusters = AgglomerativeClustering(linkage=linkage, n_clusters=n_clusters, distance_threshold=dist_threshold).fit(subsamples[['x_scaled','y_scaled']].values)

    subsamples['label'] = clusters.labels_
    subsamples['label'] = 'c'+subsamples['label'].astype(str)
    subsamples['label'].unique().shape

    subsample_mean = subsamples.groupby('label').mean()
    nbrs = NearestNeighbors(n_neighbors=1).fit(subsample_mean[['x_scaled','y_scaled']].values)
    distances, indices = nbrs.kneighbors(data[['x_scaled','y_scaled']].values)

    data['label'] = subsample_mean.iloc[indices.reshape(-1)].index.values
    
    return data




def plot_maps_and_assignments(assignments, smoothed_maps, hue_order):
#     set_rc(12,4)
    fig, axes = plt.subplots(1,3)
    titles = ['g2_vs_g4','prob_tumor','assignment']
    for idx, entry in enumerate(smoothed_maps):
        sns.heatmap(entry, vmin=-1, vmax=1, center=0, ax=axes[idx], cbar=False) 
        axes[idx].set_title(titles[idx])

    g = sns.scatterplot('x_scaled','y_scaled',data=assignments, hue='meta', legend='full', ax=axes[-1], hue_order=hue_order)
    g.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)
    axes[-1].invert_yaxis()
    axes[-1].set_title(titles[-1])
    return g 


def get_assignment_clusters(assignments, hue_order, subsample_factor=3, linkage='single', dist_threshold=0.1, n_clusters=None):
    """
    Assumes that we will rescale the x,y distance to be [0,1]
    """
    clusters = []
    for idx, label in enumerate(hue_order):
        try:
            temp_subset = assignments.loc[assignments.meta == label]
            temp_subset = subsample_and_agg_cluster(temp_subset, num_subsamples=len(temp_subset)//subsample_factor, linkage=linkage, dist_threshold=dist_threshold, n_clusters=n_clusters)
            clusters.append(temp_subset)
        
        except Exception as e:
#             print(e)
            pass
    clusters = pd.concat(clusters)
    clusters['meta_and_label'] = clusters['meta'] +'_' + clusters['label']
    clusters.loc[clusters.meta == 'stroma','meta_and_label'] = 'stroma'
    
    return clusters

def get_cluster_subset(clusters, min_cluster_count):
    clusters['meta_and_label'] = clusters['meta'] +'_' + clusters['label']
    clusters.loc[clusters.meta == 'stroma','meta_and_label'] = 'stroma'
    passing_clusters = get_indices(clusters.loc[clusters.meta != 'stroma'].groupby('meta_and_label').x.count().sort_values(ascending=False) > min_cluster_count)
    clusters_subset = clusters.loc[clusters.meta_and_label.apply(lambda x: x in passing_clusters) | (clusters.meta == 'stroma')]
    
    return clusters_subset

def get_cluster_subset_minfrac(clusters, minfrac):
#     clusters['meta_and_label'] = clusters['meta'] +'_' + clusters['label']
#     clusters.loc[clusters.meta == 'stroma','meta_and_label'] = 'stroma'
    min_cluster_count = len(clusters.loc[clusters.meta != 'stroma']) * minfrac
#     print('min_cluster_count: ', min_cluster_count)
    passing_clusters = get_indices(clusters.loc[clusters.meta != 'stroma'].groupby('meta_and_label').x.count().sort_values(ascending=False) > min_cluster_count)
    clusters_subset = clusters.loc[clusters.meta_and_label.apply(lambda x: x in passing_clusters) | (clusters.meta == 'stroma')]
    
    return clusters_subset

# 20210104 update

HUE_ORDER = ['stroma','pred_g2','pred_intermediate_grade','pred_g4',]



def subsample_and_agg_cluster(data, num_subsamples=400, linkage='single', dist_threshold=0.05, n_clusters=None):
    subsamples = data.sample(num_subsamples)
    clusters = AgglomerativeClustering(linkage=linkage, n_clusters=n_clusters, distance_threshold=dist_threshold).fit(subsamples[['x_scaled','y_scaled']].values)

    subsamples['label'] = clusters.labels_
    subsamples['label'] = 'c'+subsamples['label'].astype(str)
    subsamples['label'].unique().shape

    subsample_mean = subsamples.groupby('label').mean()
    nbrs = NearestNeighbors(n_neighbors=1).fit(subsample_mean[['x_scaled','y_scaled']].values)
    distances, indices = nbrs.kneighbors(data[['x_scaled','y_scaled']].values)

    data['label'] = subsample_mean.iloc[indices.reshape(-1)].index.values
    
    return data


def get_assignment_clusters(assignments, hue_order, subsample_factor=3, linkage='single', dist_threshold=0.1, n_clusters=None):
    """
    Assumes that we will rescale the x,y distance to be [0,1]
    """
    clusters = []
    for idx, label in enumerate(hue_order):
        try:
            temp_subset = assignments.loc[assignments.meta == label]
            temp_subset = subsample_and_agg_cluster(temp_subset, num_subsamples=len(temp_subset)//subsample_factor, linkage=linkage, dist_threshold=dist_threshold, n_clusters=n_clusters)
            clusters.append(temp_subset)
        
        except Exception as e:
#             print(e)
            pass
    clusters = pd.concat(clusters)
    clusters['meta_and_label'] = clusters['meta'] +'_' + clusters['label']
    clusters.loc[clusters.meta == 'stroma','meta_and_label'] = 'stroma'
    
    return clusters


def make_data_objs(df, slide_id, meta_vars=['stroma', 'pred_g2', 'pred_intermediate_grade', 'pred_g4']):

    results = {}
    for meta in meta_vars:
        subset = df.loc[slide_id]
        subset = subset.loc[subset.meta == meta]
        d = oe.data(subset[['x','y']], [1,2]) #instantiate the oe data object
        c = oe.cluster(d)
        c.labels['manual'] = subset.label.str.strip('c').astype(int).values # manually set previously determined clusters
        v = oe.validation(d, c)
        results[meta] = {'data':d, 'c':c, 'v':v}
        
    return results

def run_oe_metric_eval(df, ids, metrics, entry_store):
    cluster_key = 'manual'
    agg = entry_store
    
    for slide_id in ids:
        temp_metrics = {}
#         print(slide_id)
        out = make_data_objs(df, slide_id)
        for metric in metrics:
            for key,val in out.items():
                try:
                    v = val['v']
                    output_name = v.calculate(metric, cluster_key, 'parent')
                    output = v.validation[output_name]

                except Exception as e:
                    pass

        agg[slide_id] = out
    

def unpack_oe_metrics(results_dict):
    store = []
    for slide_id, entries in results_dict.items():
        slide_agg = []
        for category, val_obj in entries.items():
            temp_score_map = {slide_id: val_obj['v'].validation}
            temp_df = pd.DataFrame.from_dict(temp_score_map).transpose()
            temp_df.columns = pd.Series(temp_df.columns).apply(lambda x: x.split('_parent_manual')[0].lower()).values
            temp_df['category'] = category
            temp_df.index.name = 'slide_id'
            slide_agg.append(temp_df)
        slide_agg = pd.concat(slide_agg)
        store.append(slide_agg)
    return pd.concat(store)


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

def describe_cluster_dist(df):
    temp = df.groupby('meta').count().iloc[:,0]
    temp = temp.drop('stroma')
    dist = temp / temp.sum()
    return dist

def describe_cleaned_dist(df):
    temp = df.groupby('meta').sum()
    temp = temp.drop('stroma')
    dist = temp / temp.sum()
    return dist

def describe_category_frac(df, group_var='meta'):
    temp = df.groupby(group_var).count().iloc[:,0]
    dist = temp / temp.sum()
    return dist

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

def get_residuals_alt(df, x=['all_category_frac', 'nonstroma_category_frac'], metric='ball_hall', fit_intercept=True):
    """
    Mod of base function to instead account for both the frac rel to all tiles and tumor only 
    """
    linreg = LinearRegression(fit_intercept=fit_intercept)
    linreg.fit(df[x], df[metric])
    residuals = df[metric] - linreg.predict(df[x])
    print(linreg.coef_, linreg.intercept_)
    return residuals


def run_adjustment(df, metrics, categories, adjustment_source_cols=['all_category_frac', 'nonstroma_category_frac'], name='hybrid', fit_intercept=True):
    results = {}
    for category in categories:
        temp_results = {}
        for metric in metric_cols:
            temp = df.loc[(slice(None), category), [metric] + adjustment_source_cols].dropna()
            temp = temp.loc[~np.isinf(temp).any(1)]
            print('\n', category, metric)
            residuals = get_residuals_alt(temp, x=adjustment_source_cols, metric=metric, fit_intercept=True)
            temp_name = metric+f'_adj_{name}'
            temp_results[temp_name] = residuals
        results[category] = temp_results
    return results

###################################################
# updated 

def get_heatmap_inputs(slide_id, df, target_var='prob_g4_not_g2',tumor_var='prob_tumor', tumor_cutoff=0.70):
    temp_df = df.loc[slide_id]
    temp_df['x'] = temp_df['x'].astype(int)
    temp_df['y'] = temp_df['y'].astype(int)
    temp = df.loc[slide_id]

    temp_var_store = np.zeros((temp_df.y.max()+1, temp_df.x.max()+1)) - 1
    temp_tumor_store = np.zeros((temp_df.y.max()+1, temp_df.x.max()+1)) - 1
    temp_filtered_tumor_store = np.zeros((temp_df.y.max()+1, temp_df.x.max()+1)) - 1

    for idx, row in temp_df.iterrows():
        temp_var_store[row.y, row.x] = row[target_var]
        temp_tumor_store[row.y, row.x] = row[tumor_var]
        if row[tumor_var] >= tumor_cutoff:
            temp_filtered_tumor_store[row.y, row.x] = row[target_var]

    out = {'tumor_raw':temp_tumor_store, 'passing_tumor':temp_filtered_tumor_store, target_var:temp_var_store}
    
    return out


def create_and_smoothen_heatmaps(slide_id, df, smoothing_fn, grade_score_factor=1, tumor_score_factor=1):
    maps = get_heatmap_inputs(slide_id, df)
    stack = torch.stack([torch.Tensor(x) for x in [maps['grade'], maps['tumor_raw']]])
    smoothed_stack = smoothing_fn(stack.unsqueeze(0))[0].numpy()
    
    temp = pd.concat([pd.DataFrame(smoothed_stack[0]).unstack(), pd.DataFrame(smoothed_stack[1]).unstack()], 1).reset_index()
    temp.columns = ['x','y','g2_vs_g4','prob_tumor']
    temp['g2_vs_g4_score'] = temp['g2_vs_g4']*grade_score_factor
    temp['tumor_score'] = temp['prob_tumor']*tumor_score_factor
    
    nonblank_temp = temp.loc[temp.g2_vs_g4 >= 0]
    nonblank_temp['x_scaled'] = nonblank_temp['x'] / nonblank_temp['x'].max()
    nonblank_temp['y_scaled'] = nonblank_temp['y'] / nonblank_temp['y'].max()
    
    return nonblank_temp, smoothed_stack, maps


def assign_category(df, lower_cutoff=0.35, upper_cutoff=0.65, tumor_cutoff=0.7):
    df.loc[df.prob_tumor < tumor_cutoff, 'meta'] = 'stroma'
    df.loc[(df.prob_tumor >= tumor_cutoff) & (df.g2_vs_g4 < lower_cutoff), 'meta'] = 'pred_g2'
    df.loc[(df.prob_tumor >= tumor_cutoff) & (df.g2_vs_g4 >= upper_cutoff), 'meta'] = 'pred_g4'
    df.loc[(df.prob_tumor >= tumor_cutoff) & (df.g2_vs_g4 >= lower_cutoff) & (
    df.g2_vs_g4 < upper_cutoff), 'meta'] = 'pred_intermediate_grade'

    return df


def assign_binary_category(df, target_var='prob_g4_not_g2', tumor_var='prob_tumor', target_var_cutoff=0.5, 
                           lower_var_name='pred_g2', upper_var_name='pred_g4', tumor_cutoff=0.7):
    df.loc[df[tumor_var] < tumor_cutoff, 'meta'] = 'stroma'
    df.loc[(df[tumor_var] >= tumor_cutoff) & (df[target_var] <= target_var_cutoff), 'meta'] = lower_var_name
    df.loc[(df[tumor_var] >= tumor_cutoff) & (df[target_var] > target_var_cutoff), 'meta'] = upper_var_name
    
    return df  


def assign_threeway_category(df, target_var='prob_g4_not_g2', tumor_var='prob_tumor', lower_var_cutoff=1/3., upper_var_cutoff=2/3.,
                             lower_var_name='pred_g2', middle_var_name ='intermediate_grade', upper_var_name='pred_g4', 
                             tumor_cutoff=0.7):
    df.loc[df[tumor_var] < tumor_cutoff, 'meta'] = 'stroma'
    df.loc[(df[tumor_var] >= tumor_cutoff) & (df[target_var] <= lower_var_cutoff), 'meta'] = lower_var_name
    df.loc[(df[tumor_var] >= tumor_cutoff) & (df[target_var] > lower_var_cutoff) & (df[target_var] <= upper_var_cutoff), 'meta'] = middle_var_name
    df.loc[(df[tumor_var] >= tumor_cutoff) & (df[target_var] > upper_var_cutoff), 'meta'] = upper_var_name
    
    return df 


def subsample_and_agg_cluster(data, num_subsamples=400, linkage='single', dist_threshold=0.05, n_clusters=None,
                             x_name='x',y_name='y'):
    subsamples = data.sample(num_subsamples)
    clusters = AgglomerativeClustering(linkage=linkage, n_clusters=n_clusters, distance_threshold=dist_threshold).fit(subsamples[[x_name,y_name]].values)

    subsamples['label'] = clusters.labels_
    subsamples['label'] = 'c'+subsamples['label'].astype(str)
    subsamples['label'].unique().shape

    subsample_mean = subsamples.groupby('label').mean()
    nbrs = NearestNeighbors(n_neighbors=1).fit(subsample_mean[[x_name,y_name]].values)
    distances, indices = nbrs.kneighbors(data[[x_name,y_name]].values)

    data['label'] = subsample_mean.iloc[indices.reshape(-1)].index.values
    
    return data

def get_assignment_clusters(assignments, hue_order, subsample_factor=3, linkage='single', dist_threshold=0.1, n_clusters=None):
    """
    Assumes that we will rescale the x,y distance to be [0,1]
    """
    clusters = []
    for idx, label in enumerate(hue_order):
        try:
            temp_subset = assignments.loc[assignments.meta == label]
            temp_subset = subsample_and_agg_cluster(temp_subset, num_subsamples=len(temp_subset)//subsample_factor, linkage=linkage, dist_threshold=dist_threshold, n_clusters=n_clusters)
            clusters.append(temp_subset)
        
        except Exception as e:
            print(e)
            pass
    clusters = pd.concat(clusters)
    clusters['meta_and_label'] = clusters['meta'] +'_' + clusters['label']
    clusters.loc[clusters.meta == 'stroma','meta_and_label'] = 'stroma'
    
    return clusters

def cluster_and_evaluate(tile_df, slide_id, tumor_cutoff, linkage, dist_threshold, subsample_factor, n_clusters, meta_vars, metrics,
                         category_strategy='binary', index_name='unique_id'):

#     # assign categories
#     if category_strategy == 'binary':
#         assignments = assign_binary_category(tile_df.loc[slide_id])
#     else:
#         assignments = assign_threeway_category(tile_df.loc[slide_id])

    # run clustering
    clustering = get_assignment_clusters(tile_df, hue_order=meta_vars, 
                                             subsample_factor=subsample_factor, linkage=linkage, 
                                              dist_threshold=dist_threshold, n_clusters=n_clusters)

    # get metrics
    cluster_key = 'manual'
    metric_agg = []
    data_objs = make_data_objs(clustering, slide_id, meta_vars=meta_vars)

    for category,val in data_objs.items(): # previously had loop order swapped, resulting in len(metrics)-fold repeating of partial result stacking
        for metric in metrics:
            try:
                v = val['v']
                output_name = v.calculate(metric, cluster_key, 'parent')

            except Exception as e:
                print(e)

        if len(v.validation) > 0:
            temp_score_map = {slide_id: v.validation}
            temp_df = pd.DataFrame.from_dict(temp_score_map).transpose()
            temp_df.columns = pd.Series(temp_df.columns).apply(lambda x: x.split('_parent_manual')[0].lower()).values
            temp_df['category'] = category
            temp_df.index.name = index_name
            metric_agg.append(temp_df)
    
    return {'metrics': pd.concat(metric_agg), 'clustering':clustering}              

def plot_assignment_clusters_combined(clusters_subset, x_var='x', y_var='y', size=6, use_style=True, hue_order=None, ax=None, marker_size=5):
    """
    Assumes that we will rescale the x,y distance to be [0,1]
    """
#     set_rc(size,size)
    if use_style:
        g = sns.scatterplot(x_var,y_var, data=clusters_subset, hue='meta',style='meta_and_label', hue_order=hue_order, ax=ax, s=marker_size)
    else:
        g = sns.scatterplot(x_var,y_var, data=clusters_subset, hue='meta', hue_order=hue_order, ax=ax, s=marker_size)

    g.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)
    g.invert_yaxis()
    return g

def plot_assignment_clusters(assignments, hue_order,x_var='x', y_var='y', marker_size=5):
    """
    Assumes that we will rescale the x,y distance to be [0,1]
    """
#     set_rc(20,4)
    fig, axes = plt.subplots(1,len(hue_order))
    for idx, label in enumerate(hue_order):
        try:
            temp_subset = assignments.loc[assignments.meta == label]
            sns.scatterplot(x_var, y_var, data=temp_subset, hue='label', legend=False, ax=axes[idx], s=marker_size)
            axes[idx].set_title(label)
#             axes[idx].set_ylim(0,1)
#             axes[idx].set_xlim(0,1)
            axes[idx].invert_yaxis()
        
        except Exception as e:
            pass    

    return fig



def get_local_entropy(slide_id, df, morph_obj, target_var='prob_g4_not_g2', tumor_var='prob_tumor'):
    x = get_heatmap_inputs(slide_id, df, target_var=target_var, tumor_var=tumor_var)
    y = x['passing_tumor']
    mask = (y != -1).astype(int)
    entr_img = local_entropy(y, morph_obj, mask=mask)
    masked = np.ma.masked_where(mask == 0, entr_img)
    
    return masked.compressed(), masked.filled(0)