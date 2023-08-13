import sys
import pandas as pd
import seaborn as sns

import scanpy as sc
sc.settings.verbosity = 3 
import anndata

import dask
from dask.distributed import Client
import timeit
from glob import glob
from sklearn.mixture import GaussianMixture
from skimage.segmentation import mark_boundaries
from skimage.color import label2rgb
import skimage.io as io
from skimage.io import imread
import os
import pandas as pd
import tifffile
from tifffile import TiffFile
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.cluster import AgglomerativeClustering
import parse
from matplotlib.colors import to_rgb
import plotly.express as px

import xarray

from anndata import AnnData
from statannotations.Annotator import Annotator
from itertools import product, combinations, permutations
from scipy.stats import mannwhitneyu, fisher_exact, ttest_ind

import squidpy as sq


from sklearn.linear_model import LinearRegression



def add_log10_density(df, col):
    df[f'log10({col} + 1)'] = np.log10(df[col] + 1)
    
    
def flex_save(path, dpi=800, extensions=['.png','.pdf']):
    for ext in extensions:
        plt.savefig(path+ext, dpi=dpi, bbox_inches='tight')


def regress_out(df, x, y, fit_intercept=True):
    linreg = LinearRegression(fit_intercept=fit_intercept)
    linreg.fit(df[[x]], y=df[[y]])
    residuals = linreg.predict(df[[x]])
    df[f'pred_{y}_cond_{x}'] = residuals
    df[f'residual_{y}_cond_{x}'] = df[y] - df[f'pred_{y}_cond_{x}']
    

def make_statannotations_pairs(i, j, mode='inner'):
    if mode == 'inner':
        pairs = []
        for x in i:
            pairs.extend(list(combinations(product([x],j), 2)))
            
    if mode == 'full':
         pairs = list(combinations(product(i,j),2))
            
    return pairs


def get_separated_heatmaps_simplified(plotting_df, col):
    x_max, y_max = plotting_df[['x','y']].max()
    placeholder = np.ones((y_max+1, x_max+1))*-1
    for (x,y), row in plotting_df.set_index(['x','y']).iterrows():
        placeholder[y,x] = row[col]

    return placeholder


def create_spatial_anndata(df, feature_cols):
    
    coord = df[['cell_x_harmonized','cell_y_harmonized']].values
    adata = AnnData(X=df[feature_cols], obs=df.drop(columns=feature_cols), obsm={"spatial": coord})
    convert_adata_str_to_cat(adata)
    
    return adata


def run_squidypy_processing(adata, neighbor_strategy='delaunay', nn=None, radius=None, cell_type_col='detailed_cell_type', n_jobs=None,
                           run_neighborhood_enrichment=True, run_cooccurrence=False, run_centrality_scores=False):
    # nearest neighbor construction
    if neighbor_strategy == 'delaunay':
        print('delaunay')
        sq.gr.spatial_neighbors(adata, coord_type='generic', delaunay=True, radius=radius)
    else:
        print(f'Spatial neighbors with NN={nn}')
        sq.gr.spatial_neighbors(adata, coord_type='generic', n_neighs=nn, radius=radius)

    # neighborhood enrichment
    if run_neighborhood_enrichment:
        sq.gr.nhood_enrichment(adata, cluster_key=cell_type_col, n_jobs=n_jobs)
        
    # co-occurrence
    if run_cooccurrence:
        sq.gr.co_occurrence(adata, cluster_key=cell_type_col, n_jobs=n_jobs)
    
    # node centralities per cluster 
    if run_centrality_scores:
        sq.gr.centrality_scores(adata, cluster_key=cell_type_col, n_jobs=n_jobs)
        
        

def filter_adata_by_fov(adata, xmin, xmax, ymin, ymax):
    crit = (adata.obs['fov_x'] > xmin) & (adata.obs['fov_x'] < xmax) & (adata.obs['fov_y'] > ymin) & (adata.obs['fov_y'] < ymax)
    print(crit.sum())
    return adata[crit]

def filter_df_by_fov(df, xmin, xmax, ymin, ymax):
    crit = (df['fov_x'] > xmin) & (df['fov_x'] < xmax) & (df['fov_y'] > ymin) & (df['fov_y'] < ymax)
    print(crit.sum())
    return df.loc[crit]

def convert_adata_str_to_cat(adata):
    for col in adata.obs:
        if type(adata.obs[col].values[0]) == str:
            # print(f'converting {col} to category') 
            adata.obs[col] = adata.obs[col].astype('category')


def get_labels_string(annotation, sep=' vs. '):
    labels_string = sep.join([struct["label"]
                          for struct in annotation.structs])    
    return labels_string

def retrieve_label_and_pval_from_annotation(annotation):
    label = get_labels_string(annotation)
    pval = annotation.data.pvalue
    
    return label, pval

def unpack_annotator(annotator):
    results = {}
    for annotation in annotator.annotations:
        label = get_labels_string(annotation)
        pval = annotation.data.pvalue
        results[label] = pval
    return results




def move_legend(g, loc='center', xy=(1.5,0.5)):
    sns.move_legend(g, loc=loc, bbox_to_anchor=xy)

def add_af_ratios(df, cols):
    for col in cols:
        df[f'{col}_af_ratio'] = df[col]/df['autofluorescence']
        
def add_raw_channel_values(df, cols):
    for col in cols:
        df[f'{col}_raw'] = invert_arcinsh_norm(df[col])


def double_check_gates(cutoffs, plot_data, size=400):
    checked_cutoffs = {}
    
    ################## CD8 GATE ##################
    marker = 'cd8'

    gate_name = 'cd8_gate'
    print(gate_name)
    fig = px.histogram(plot_data.reset_index(), x=marker, width=size,  height=size, nbins=200, opacity=0.15)
    fig.add_vline(cutoffs[gate_name])
    fig.show()
    
    check = input(f'{gate_name} OK?')
    if check == 'y':
        checked_cutoffs[gate_name] = cutoffs[gate_name]
    else:
        checked_cutoffs[gate_name] = float(check)
        print(f'Updated {gate_name}')

    ################## TUMOR GATE ##################
    crit = plot_data['cd8'] <= cutoffs['cd8_gate']
    subset = plot_data.loc[crit]
    marker = 'tumor_specific'

    gate_name = 'tumor_gate'
    fig = px.histogram(subset.reset_index(), x=marker, width=size,  height=size, nbins=200, opacity=0.15)
    fig.add_vline(cutoffs[gate_name])
    fig.show()
    
    check = input(f'{gate_name} OK?')
    if check == 'y':
        checked_cutoffs[gate_name] = cutoffs[gate_name]
    else:
        checked_cutoffs[gate_name] = float(check)
        print(f'Updated {gate_name}')
        

    ################## CD8+ AF GATE ##################
    crit = (plot_data['cd8'] > cutoffs['cd8_gate']) & (plot_data['tumor_specific'] <= cutoffs['tumor_gate'])
    subset = plot_data.loc[crit]
    marker = 'cd8_af_ratio'
    
    gate_name = 'cd8_to_af_ratio_within_cd8_gate'
    fig = px.histogram(subset.reset_index(),x=marker, width=size,  height=size, nbins=200, opacity=0.15)
    fig.add_vline(cutoffs[gate_name])
    fig.show()

    check = input(f'{gate_name} OK?')
    if check == 'y':
        checked_cutoffs[gate_name] = cutoffs[gate_name]
    else:
        checked_cutoffs[gate_name] = float(check)
        print(f'Updated {gate_name}')

    ################## CD8+ AF- PD1 GATE ##################
    crit = (plot_data['cd8'] > cutoffs['cd8_gate']) & (plot_data['cd8_af_ratio'] > cutoffs['cd8_to_af_ratio_within_cd8_gate'])
    crit = crit & (plot_data['tumor_specific'] <= cutoffs['tumor_gate'])

    subset = plot_data.loc[crit]
    marker = 'pd1'
    
    gate_name = 'pd1_within_cd8_gate'
    fig = px.histogram(subset.reset_index(),x=marker, width=size,  height=size, nbins=200, opacity=0.15)
    fig.add_vline(cutoffs[gate_name])
    fig.show()

    check = input(f'{gate_name} OK?')
    if check == 'y':
        checked_cutoffs[gate_name] = cutoffs[gate_name]
    else:
        checked_cutoffs[gate_name] = float(check)
        print(f'Updated {gate_name}')

    ################## OTHER FOXP3 GATE ##################
    crit = (plot_data['cd8'] <= cutoffs['cd8_gate']) & (plot_data['tumor_specific'] <= cutoffs['tumor_gate'])
    subset = plot_data.loc[crit]
    marker = 'foxp3'
    
    gate_name = 'other_foxp3_gate'
    fig = px.histogram(subset.reset_index(), x=marker, width=size,  height=size, nbins=200, opacity=0.15)
    fig.add_vline(cutoffs[gate_name])
    fig.show()

    check = input(f'{gate_name} OK?')
    if check == 'y':
        checked_cutoffs[gate_name] = cutoffs[gate_name]
    else:
        checked_cutoffs[gate_name] = float(check)
        print(f'Updated {gate_name}')


    ################## TUMOR+ PDL1 GATE ##################
    crit = (plot_data['tumor_specific'] > cutoffs['tumor_gate'])
    subset = plot_data.loc[crit]
    marker = 'pdl1'
    
    gate_name = 'tumor_pdl1_gate'
    fig = px.histogram(subset.reset_index(), x=marker, width=size,  height=size, nbins=200, opacity=0.15)
    fig.add_vline(cutoffs[gate_name])
    fig.show()
    
    check = input(f'{gate_name} OK?')
    if check == 'y':
        checked_cutoffs[gate_name] = cutoffs[gate_name]
    else:
        checked_cutoffs[gate_name] = float(check)
        print(f'Updated {gate_name}')
        
    return checked_cutoffs

def vis_gates(cutoffs, plot_data, xmin=6, xmax=10):
    fig,axes = plt.subplots(1,6, figsize=[20,3])
    ################## CD8 GATE ##################
    marker = 'cd8'

    gate_name = 'cd8_gate'
    print(gate_name)
    sns.histplot(ax=axes[0], data=plot_data.reset_index(), x=marker, bins=200)
    axes[0].axvline(cutoffs[gate_name], c='r')
    axes[0].set_xlim(xmin, xmax)


    ################## TUMOR GATE ##################
    crit = plot_data['cd8'] <= cutoffs['cd8_gate']
    subset = plot_data.loc[crit]
    marker = 'tumor_specific'

    gate_name = 'tumor_gate'
    sns.histplot(ax=axes[1], data=subset.reset_index(), x=marker, bins=200)
    axes[1].axvline(cutoffs[gate_name], c='r')
    axes[1].set_xlim(xmin, xmax)


    ################## CD8+ AF GATE ##################
    crit = (plot_data['cd8'] > cutoffs['cd8_gate']) & (plot_data['tumor_specific'] <= cutoffs['tumor_gate'])
    subset = plot_data.loc[crit]
    marker = 'cd8_af_ratio'
    
    gate_name = 'cd8_to_af_ratio_within_cd8_gate'
    sns.histplot(ax=axes[2], data=subset.reset_index(), x=marker, bins=200)
    axes[2].axvline(cutoffs[gate_name], c='r')
    # axes[2].set_xlim(0.8, 1.5)
    

    ################## CD8+ AF- PD1 GATE ##################
    crit = (plot_data['cd8'] > cutoffs['cd8_gate']) & (plot_data['cd8_af_ratio'] > cutoffs['cd8_to_af_ratio_within_cd8_gate'])  # error originally via wrong channel usage!!
    crit = crit & (plot_data['tumor_specific'] <= cutoffs['tumor_gate'])

    subset = plot_data.loc[crit]
    marker = 'pd1'
    
    gate_name = 'pd1_within_cd8_gate'
    sns.histplot(ax=axes[3], data=subset.reset_index(), x=marker, bins=200)
    axes[3].axvline(cutoffs[gate_name], c='r')
    axes[3].set_xlim(xmin, xmax)

    ################## OTHER FOXP3 GATE ##################
    crit = (plot_data['cd8'] <= cutoffs['cd8_gate']) & (plot_data['tumor_specific'] <= cutoffs['tumor_gate'])
    subset = plot_data.loc[crit]
    marker = 'foxp3'
    
    gate_name = 'other_foxp3_gate'
    sns.histplot(ax=axes[4], data=subset.reset_index(), x=marker, bins=200)
    axes[4].axvline(cutoffs[gate_name], c='r')
    axes[4].set_xlim(xmin, xmax)

    ################## TUMOR+ PDL1 GATE ##################
    crit = (plot_data['tumor_specific'] > cutoffs['tumor_gate'])
    subset = plot_data.loc[crit]
    marker = 'pdl1'
    
    gate_name = 'tumor_pdl1_gate'
    sns.histplot(ax=axes[5], data=subset.reset_index(), x=marker, bins=200)
    axes[5].axvline(cutoffs[gate_name], c='r')
    axes[5].set_xlim(xmin, xmax)

    plt.show()

    
    
def load_fov_data(df):
    assert df.index.name == 'fov'
    
    seg = [] 
    data = []
    
    for fov_idx, row in df.iterrows():
        seg.append(xarray.load_dataarray(row['seg']))
        data.append(xarray.load_dataarray(row['data']))
    
    seg = np.stack(seg, 0)
    data = np.stack(data, 0)
    
    return seg, data

def quantile_normalize_fov_examples(data, channels=[2,3,0],lower_clip_percentile=25, upper_clip_percentile=99):
    store = []
    for color_idx, channel_idx in enumerate(channels):
            clip_min, clip_max = np.percentile(data[...,[channel_idx]], (lower_clip_percentile,upper_clip_percentile))
            clipped = np.clip(data[...,[channel_idx]], 0, clip_max)
            clipped = (clipped - clip_min)/clip_max
            clipped = np.clip(clipped, 0, 1)
            store.append(clipped[...,0])

    return np.stack(store,-1)
    

def visualize_fov_samples(case_id, cell_df, tumor_parses, named_cutoffs, n_samples=5, width=7, height=15, channels=[2, 3, 0], 
                          lower_clip_percentile=25, upper_clip_percentile=99, max_cell_area=2000,
                          cell_label_col='cell_label',
                          mask_key = {0: 'cd8+', 1: 'other', 2: 'tumor+', 3: 'ungated/omit'},
                          full_color_map= {idx:color for idx,color in enumerate(['black','green','grey','purple','yellow','orange',])},
                          return_samples=False
                         ):

    print(mask_key, full_color_map)

    parse_samples = tumor_parses.loc[case_id].sample(n_samples)

    seg, data = load_fov_data(parse_samples)
    store = quantile_normalize_fov_examples(data, channels=channels, lower_clip_percentile=lower_clip_percentile, upper_clip_percentile=upper_clip_percentile)


    for cutoff_set in named_cutoffs:
        print(cutoff_set)
        set_rc(int(n_samples*width), height)
        fig, axes = plt.subplots(2,n_samples)
    
        for i, (fov_idx, row) in enumerate(parse_samples.iterrows()):
            # plt.suptitle('Red: DAPI \n Green: CD8 \n Blue: Tumor')
            axes[0,i].imshow(mark_boundaries(store[i], seg[i,...,0].astype(int), color=(0.5, 0.5, 0.5)))
            # axes[0,i].imshow(store)


            tile = np.copy(seg[i,...,0])
            fill_mask = np.copy(seg[i,...,0])
            print(fill_mask.dtype)

            fov_calls = cell_df.loc[(case_id, fov_idx)]


            apply_manual_cell_calling(cutoff_set, fov_calls)

            # print('Dropping big cells (1500 area)')
            fov_calls = fov_calls.loc[fov_calls['area'] < max_cell_area]

            tile_img = ((tile > 0)*255).astype(np.uint8)
            print(fov_calls['cell_label'].value_counts())
            for cell_type_idx, cell_type in mask_key.items():
                print(cell_type_idx, cell_type)
                temp_mask = np.isin(tile, fov_calls.loc[fov_calls[cell_label_col] == cell_type,'label'].values)
                fill_mask[temp_mask] = cell_type_idx+1


            # catch cells we dropped and mark as orange
            fill_mask[fill_mask > 4] = 5

            # plot_colors = [full_color_map[k] for k in np.unique(fill_mask.astype(int))]
            
            # 0.12 --> 0.14+ changed behavior to assume not giving background its own color
            plot_colors = [full_color_map[k] for k in np.unique(fill_mask.astype(int))[1:]]  
            
            print(plot_colors)
            colormapped = label2rgb(fill_mask.astype(int), tile_img, alpha=1., colors=plot_colors)
            axes[1,i].imshow(mark_boundaries(colormapped,tile.astype(int), color=(0,0,0)))
        plt.show()
        
    if return_samples:
        return parse_samples, seg, data, fill_mask, fov_calls
        
        
def make_suggestions(global_data, subset_data, marker):
    sugg = run_gating_flex(global_data.reset_index(), marker, 2)
    print(f'2-component global gate suggestion: {sugg}', )
    print('===================')
    sugg = run_gating_flex(global_data.reset_index(), marker, 3)
    print(f'3-component global gate suggestion: {sugg}', )
    print('===================')

    if subset_data is not None:
        sugg = run_gating_flex(subset_data.reset_index(), marker, 2)
        print(f'2-component subset gate suggestion: {sugg}', )
        print('===================')

        sugg = run_gating_flex(subset_data.reset_index(), marker, 3)
        print(f'3-component subset gate suggestion: {sugg}', )
        print('===================')


def assign_gates(plot_data, size=600):
    cutoffs = {}

    ################## CD8 GATE ##################
    marker = 'cd8'
    fig = px.histogram(plot_data.reset_index(), x=marker, width=size,  height=size, nbins=200, opacity=0.15)
    fig.show()
    make_suggestions(plot_data, None, marker)
    cutoff = float(input('CD8 thresh'))
    cutoffs['cd8_gate'] = cutoff

    ################## TUMOR GATE ##################
    crit = plot_data['cd8'] <= cutoffs['cd8_gate']
    subset = plot_data.loc[crit]
    marker = 'tumor_specific'
    make_suggestions(plot_data, subset, marker)
    fig = px.histogram(subset.reset_index(), x=marker, width=size,  height=size, nbins=200, opacity=0.15)
    fig.show()
    cutoff = float(input('Tumor thresh'))
    cutoffs['tumor_gate'] = cutoff



    ################## CD8+ AF GATE ##################
    crit = (plot_data['cd8'] > cutoffs['cd8_gate']) & (plot_data['tumor_specific'] <= cutoffs['tumor_gate'])
    subset = plot_data.loc[crit]
    marker = 'cd8_af_ratio'
    make_suggestions(plot_data, subset, marker)
    fig = px.histogram(subset.reset_index(),x=marker, width=size,  height=size, nbins=200, opacity=0.15)
    fig.show()
    cutoff = float(input('CD8:AF Ratio thresh within CD8+'))
    cutoffs['cd8_to_af_ratio_within_cd8_gate'] = cutoff

    ################## CD8+ AF- PD1 GATE ##################
    crit = (plot_data['cd8'] > cutoffs['cd8_gate']) & (plot_data['autofluorescence'] > cutoffs['cd8_to_af_ratio_within_cd8_gate'])
    crit = crit & (plot_data['tumor_specific'] <= cutoffs['tumor_gate'])

    subset = plot_data.loc[crit]
    marker = 'pd1'
    make_suggestions(plot_data, subset, marker)
    fig = px.histogram(subset.reset_index(),x=marker, width=size,  height=size, nbins=200, opacity=0.15)
    fig.show()
    cutoff = float(input('PD1 thresh within CD8+ AF-'))
    cutoffs['pd1_within_cd8_gate'] = cutoff



    ################## OTHER FOXP3 GATE ##################
    crit = (plot_data['cd8'] <= cutoffs['cd8_gate']) & (plot_data['tumor_specific'] <= cutoffs['tumor_gate'])
    subset = plot_data.loc[crit]
    marker = 'foxp3'
    make_suggestions(plot_data, subset, marker)
    fig = px.histogram(subset.reset_index(), x=marker, width=size,  height=size, nbins=200, opacity=0.15)
    fig.show()
    cutoff = float(input('Other+ FOXP3 thresh'))
    cutoffs['other_foxp3_gate'] = cutoff



    ################## TUMOR+ PDL1 GATE ##################
    crit = (plot_data['tumor_specific'] > cutoffs['tumor_gate'])
    subset = plot_data.loc[crit]
    marker = 'pdl1'
    make_suggestions(plot_data, subset, marker)

    fig = px.histogram(subset.reset_index(), x=marker, width=size,  height=size, nbins=200, opacity=0.15)
    fig.show()

    cutoff = float(input('Tumor+ PDL1 thresh'))
    cutoffs['tumor_pdl1_gate'] = cutoff
    
    return cutoffs


def apply_manual_cell_calling(named_cutoffs, df):

    if 'af_gate' in named_cutoffs.keys():
        ################## CD8+ AF- PD1 GATE ##################
        crit = (df['cd8'] > named_cutoffs['cd8_gate']) & (df['cd8_af_ratio'] > named_cutoffs['cd8_to_af_ratio_global'])
        crit = crit & (df['tumor_specific'] <= named_cutoffs['tumor_gate'])
        df.loc[crit & (df['pd1'] > named_cutoffs['pd1_within_cd8_gate'] ), 'manual_cell_call'] = 'CD8+ PD1+'
        df.loc[crit & (df['pd1'] <= named_cutoffs['pd1_within_cd8_gate'] ), 'manual_cell_call'] = 'CD8+ PD1-'



        ################## OTHER FOXP3 GATE ##################
        crit = (df['cd8'] <= named_cutoffs['cd8_gate']) & (df['tumor_specific'] <= named_cutoffs['tumor_gate'])
        df.loc[crit & (df['foxp3'] > named_cutoffs['other_foxp3_gate'] ), 'manual_cell_call'] = 'Other FOXP3+'
        df.loc[crit & (df['foxp3'] <= named_cutoffs['other_foxp3_gate'] ), 'manual_cell_call'] = 'Other FOXP3-'


        ################## TUMOR+ PDL1 GATE ##################
        crit = (df['tumor_specific'] > named_cutoffs['tumor_gate']) & (df['cd8'] <= named_cutoffs['cd8_gate'])
        df.loc[crit & (df['pdl1'] > named_cutoffs['tumor_pdl1_gate'] ), 'manual_cell_call'] = 'Tumor+ PDL1+'
        df.loc[crit & (df['pdl1'] <= named_cutoffs['tumor_pdl1_gate'] ), 'manual_cell_call'] = 'Tumor+ PDL1-'

        ################## HIGH AF GLOBAL REMOVAL ##################
        crit = (df['autofluorescence'] > named_cutoffs['af_gate']) 
        df.loc[crit, 'manual_cell_call'] = 'Omit'

        ################## LOW CD8 to AF REMOVAL ##################
        crit = (df['cd8_af_ratio'] <= named_cutoffs['cd8_to_af_ratio_global']) & (df['cd8'] > named_cutoffs['cd8_gate'])
        df.loc[crit, 'manual_cell_call'] = 'Omit'

        df['manual_cell_call'] = df['manual_cell_call'].fillna('Ungated')
        
    else:
        ################## CD8+ AF- PD1 GATE ##################
        crit = (df['cd8'] > named_cutoffs['cd8_gate']) & (df['cd8_af_ratio'] > named_cutoffs['cd8_to_af_ratio_within_cd8_gate'])
        crit = crit & (df['tumor_specific'] <= named_cutoffs['tumor_gate'])
        df.loc[crit & (df['pd1'] > named_cutoffs['pd1_within_cd8_gate'] ), 'manual_cell_call'] = 'CD8+ PD1+'
        df.loc[crit & (df['pd1'] <= named_cutoffs['pd1_within_cd8_gate'] ), 'manual_cell_call'] = 'CD8+ PD1-'



        ################## OTHER FOXP3 GATE ##################
        crit = (df['cd8'] <= named_cutoffs['cd8_gate']) & (df['tumor_specific'] <= named_cutoffs['tumor_gate'])
        df.loc[crit & (df['foxp3'] > named_cutoffs['other_foxp3_gate'] ), 'manual_cell_call'] = 'Other FOXP3+'
        df.loc[crit & (df['foxp3'] <= named_cutoffs['other_foxp3_gate'] ), 'manual_cell_call'] = 'Other FOXP3-'


        ################## TUMOR+ PDL1 GATE ##################
        crit = (df['tumor_specific'] > named_cutoffs['tumor_gate']) & (df['cd8'] <= named_cutoffs['cd8_gate'])
        df.loc[crit & (df['pdl1'] > named_cutoffs['tumor_pdl1_gate'] ), 'manual_cell_call'] = 'Tumor+ PDL1+'
        df.loc[crit & (df['pdl1'] <= named_cutoffs['tumor_pdl1_gate'] ), 'manual_cell_call'] = 'Tumor+ PDL1-'


        ################## LOW CD8 to AF REMOVAL ##################
        crit = (df['cd8_af_ratio'] <= named_cutoffs['cd8_to_af_ratio_within_cd8_gate']) & (df['cd8'] > named_cutoffs['cd8_gate'])
        df.loc[crit, 'manual_cell_call'] = 'Omit'

        df['manual_cell_call'] = df['manual_cell_call'].fillna('Ungated')
    
    #######################################################################################
    #### simple 4-way class
    df['cell_label'] = np.nan # catch prior anno and flag
    crit = df['manual_cell_call'].str.startswith('CD8+')
    df.loc[crit, 'cell_label'] = 'cd8+'

    crit = df['manual_cell_call'].str.startswith('Tumor+')
    df.loc[crit, 'cell_label'] = 'tumor+'

    crit = df['manual_cell_call'].str.startswith('Other')
    df.loc[crit, 'cell_label'] = 'other'

    df['cell_label'] = df['cell_label'].fillna('ungated/omit')


def run_gating(data, col, col_name=None):
    if col_name is None:
        col_name = f'gmm_{col}'
        
    preds = GaussianMixture(n_components=2).fit_predict(data[[col]].values)
    data[col_name] = preds

    group_mins = data.groupby([col_name])[col].min()
    # print(group_mins)
    cutoff = group_mins.max()
    return cutoff


def run_gating_flex(data, col, n_components=2, col_name=None):
    if col_name is None:
        col_name = f'gmm_{col}'
        
    preds = GaussianMixture(n_components=n_components).fit_predict(data[[col]].values)
    data[col_name] = preds

    group_mins = data.groupby([col_name])[col].min()
    # print(group_mins)
    # print(data[col_name].value_counts())
    cutoff = group_mins.max()
    return cutoff



def format_parses(template, parses, parse_name='filepath'):
    parses = pd.DataFrame({x:parse.parse(template, x).named for x in parses}).transpose()
    parses.index.name = parse_name
    return parses.reset_index()


def run_subclustering_multi_v2(
    adata_path, 
    rel_tumor_cutoff = 0.9, 
    louvain_resolutions=[0.2, 1.],
):
    print(f'Loading adata for {adata_path}...')
    adata = sc.read_h5ad(adata_path) 
    print('Done!')
    adata.obs_names_make_unique()

    df = adata.obs.reset_index()
    print('Collapsing Louvain Clusters on Median Tumor Marker Expression')
    df = collapse_clusters(df, 'tumor_specific')
    cluster_max_idx = df.groupby(['agg_cluster_tumor_specific']).tumor_specific.mean().idxmax()
    df['tumor_louvain'] = df['agg_cluster_tumor_specific'] == cluster_max_idx
    
    # NEED to have these connected via adata.obs -- doesn't matter if index of obs df matches external df...
    adata.obs = adata.obs.join(df.set_index('case_id')['tumor_louvain'])
    nontumor_adata = adata[~adata.obs['tumor_louvain']]
    tumor_adata = adata[adata.obs['tumor_louvain']]
    
    perc_tumor = df['tumor_louvain'].mean()
    print(f'Done! Tumor Fraction = {perc_tumor :.2f}')
    
    print(f'Non-tumor: {nontumor_adata.obs.shape}')
    print(f'Tumor: {tumor_adata.obs.shape}')
    
    print('Running non-tumor subclustering')
    for res in louvain_resolutions:
        print(res)
        key_name = f'louvain_nontumor_res={res}'
        nontumor_adata = recluster(nontumor_adata, resolution=res, key_added=key_name)
        tumor_adata.obs[key_name] = np.nan
        
    print('Running tumor subclustering')
    for res in louvain_resolutions:
        print(res)
        key_name = f'louvain_tumor_res={res}'
        tumor_adata = recluster(tumor_adata, resolution=res, key_added=key_name)
        nontumor_adata.obs[key_name] = np.nan        
        
    print('Done! Collapsing adata.obs for each compartment')
    tumor_df = tumor_adata.obs.reset_index()
    add_umap_coord(tumor_adata, tumor_df)
    tumor_df['compartment'] = 'tumor'

    nontumor_df = nontumor_adata.obs.reset_index()
    add_umap_coord(nontumor_adata, nontumor_df)
    nontumor_df['compartment'] = 'nontumor'

    combined_df = pd.concat([tumor_df, nontumor_df])
    
    return combined_df


def recluster(adata, resolution=0.4, use_rapids=True, key_added='louvain'):
    if use_rapids:
        print('using rapids GPU implementation')
        method='rapids'
        flavor='rapids'
    else:
        method='umap'
        flavor='vtraag'
        
    recluster_adata = adata.copy()
    sc.tl.louvain(recluster_adata, resolution=resolution, flavor=flavor, key_added=key_added)

    return recluster_adata


def invert_arcinsh_norm(data, linear_factor=100):
    return np.sinh(data)/linear_factor


def run_arcsinh_norm(data, linear_factor=100):
    return np.arcsinh(data*100)


def get_indices(series):
    return [idx for idx, x in series.iteritems() if x]


def reset_set_idx(df, idx_name):
    return df.reset_index().set_index(idx_name)


def rotate_labels(g, rotation=45):
    g.set_xticklabels(rotation=rotation)

    
def add_umap_coord(adata, df):
    df[['umap1','umap2']] = adata.obsm['X_umap']
 

def split_compartments(
    adata_path, 
    rel_tumor_cutoff = 0.9, 
):
    print(f'Loading adata for {adata_path}...')
    adata = sc.read_h5ad(adata_path) 
    print('Done!')
    adata.obs_names_make_unique()

    # means = invert_arcinsh_norm(adata.obs.groupby('louvain').mean())
    means = invert_arcinsh_norm(adata.obs.groupby('louvain').median())
    
    print('Finding tumor clusters...')
    # tumor cluster filtering
    t_max = means['tumor_specific'].max()
    rel_t = means['tumor_specific']/t_max
    print(rel_t.sort_values())
    
    tumor_clusters = rel_t.loc[(rel_t >= rel_tumor_cutoff)].index.values
    
    df = adata.obs.reset_index().set_index('louvain')
    df['tumor_louvain'] = False
    df.loc[tumor_clusters, 'tumor_louvain'] = True
    
    return df


def assign_simple_cell_classes_v2(df):
    crit = df['tumor_gate'] & ~df['cd8_gate']
    df.loc[crit, 'manual_class_simple'] = 'tumor'

    crit = (~df['tumor_gate']) & (df['cd8_gate'])
    df.loc[crit, 'manual_class_simple'] = 'cd8'
    
    crit = (df['tumor_gate']) & (df['cd8_gate'])
    df.loc[crit, 'manual_class_simple'] = 'other_double_pos'
    
    crit = (~df['tumor_gate']) & (~df['cd8_gate'])
    df.loc[crit, 'manual_class_simple'] = 'other_double_neg'

    
def collapse_clusters(df, marker, cluster_col='louvain', n_components=2):
    clusterer = AgglomerativeClustering(n_clusters=n_components)
    cluster_med = df.groupby(cluster_col)[marker].median()
    
    preds = clusterer.fit_predict(cluster_med.values.reshape(-1,1))
    preds = pd.Series(preds, index=cluster_med.index, name=f'agg_cluster_{marker}')
    out = df.set_index(cluster_col).join(preds).reset_index()
    
    return out


def set_rc(x=10,y=10, font_scale=1., style='white', font='Arial', palette='colorblind', **kwargs):
    sns.set(rc={'figure.figsize':[x,y]}, font_scale=font_scale, style=style, font=font, palette='colorblind', **kwargs)
    


qptiff_channels = ['dapi', 'foxp3', 'tumor_specific', 'cd8', 'pd1', 'pdl1','autofluorescence']

