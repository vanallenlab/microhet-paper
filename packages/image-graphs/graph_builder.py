#Use pip to install the following packages in CLI to the appropriate conda environment/kernel before running 

#For running hovernet nucleus segmentation
# !pip install scanpy
# !pip install torchvision
# !pip install opencv-python

#For running graph construction
# !pip install pyflann
# !pip install networkx
# !pip install torch_sparse, torch_scatter
# !pip install git+https://github.com/rusty1s/pytorch_geometric.git

#General imports 
import numpy as np
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
from torch.optim.lr_scheduler import StepLR
from PIL import Image
import cv2
import skimage 
from torchvision import models, transforms
import itertools
import math, random
import pandas as pd
import seaborn as sns
import scanpy as sc
from glob import glob
import sys,os

#For hovernet
import albumentations as A
from pathml.datasets.pannuke import PanNukeDataModule
from pathml.ml.hovernet import HoVerNet, loss_hovernet, post_process_batch_hovernet, _HoverNetDecoder
from pathml.ml.utils import wrap_transform_multichannel, dice_score
from pathml.utils import plot_segmentation

#For graph construction
from collections import OrderedDict
from pyflann import *
import skimage.feature
import networkx as nx
import torchvision.transforms.functional as F
import torch_geometric.data as data
import torch_geometric.utils as utils
import torch_geometric

def make_model():
    """Prerequisite code block - instantiates the 'device' and hovernet model 
    for all future inference.
    
    :return: device (GPU device under a name variable), 
    hovernet (inference-ready pytorch nucleus segmentation model)
    """
    #Prerequisite code block
    #prepare the model, the GPU, and send the model to the GPU
    device = torch.device("cuda:0")
    checkpoint = torch.load("/mnt/disks/prostate_data/hovernet/hovernet_pannuke.pt", map_location='cpu')

    n_classes_pannuke = 6

    hovernet = HoVerNet(n_classes=n_classes_pannuke) #nuclei will be classified into 1 of 6 classes after segmentation
    hovernet = torch.nn.DataParallel(hovernet) # wrap model to use multi-GPU
    hovernet.load_state_dict(checkpoint) #load the best checkpoint for prediction/finetuning 

    hovernet = hovernet.module
    hovernet.to(device);
    hovernet.eval();
    return device, hovernet

def pil_loader(path):
    """
    Open single image as file to avoid ResourceWarning 
    (https://github.com/python-pillow/Pillow/issues/835).
    (For segmentation purposes; this function is rarely called alone.)
    
    :param path: string path name
    
    :return: PIL image object.
    """
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
    
def as_array(path):
    """Open single image as np array.
    
    :param path: string path name
    
    :return: np array.
    """
    x = np.asarray(pil_loader(path))
    return x 

def get_cell_image(img, cx, cy, size=512):
    """
    Extract a "context-window" around a specified nucleus of size 64x64px. 
    
    :param img: np array of dimensions (size, size, 3), 
    :param cx: x coordinate of nucleus centroid
    :param cy: y coordinate of nucleus centroid
    :param size: (optional) int for img dimensions (default 512; change 
    to match the size of input tiles)
    
    :return: image array. If function is called on a grayscale array, 
    return dimension is (64, 64); otherwise, (64, 64, 3).
    """
    cx = 32 if cx < 32 else size - 32 if cx > size - 32 else cx
    cy = 32 if cy < 32 else size - 32 if cy > size - 32 else cy
    if len(img.shape) == 3:
        return img[cy - 32:cy + 32, cx - 32:cx + 32, :]
    else:
        return img[cy - 32:cy + 32, cx - 32:cx + 32]
    
def get_mean_contour_intensity_grayscale(gray_img, contours):
    """
    Proxy for how dark/light the pixels are within the segmented nucleus area.
    
    :param gray_img: cv2-converted grayscale image of segmented nucleus area
    :param contours: contour (list of 2-tuples) produced from nucleus segmentation
    
    :return: mean pixel intensity over nucleus contour area
    """
    img_size = gray_img.shape[0]
    assert gray_img.shape[0] == gray_img.shape[1]
    nucl_mask = np.zeros((img_size, img_size))
    cv2.fillPoly(nucl_mask, pts=contours, color=(1.));  # use contour area to make a mask
    z = np.ma.masked_array(data=gray_img,
                           mask=(nucl_mask != 1.))  # mask masked version to select out only the contour area
    return z.compressed().mean()  # take mean grayscale pixel intensity over contour area

def get_basic_cell_features(img, grayscale, contour):
    """
    Modified feature extractor for single nucleus basic (non-neural) features.
    
    :param img: np array image (dimensions (tile_dim, tile_dim, 3))
    :param grayscale: grayscale version of np array image (dimensions (tile_dim, tile_dim))
    :param contour: contour produced from nucleus segmentation (list of (x,y) 
    coordinates for one nucleus)
    
    :return: x coordinate, y coordinate of nucleus centroid, concatenated
    feature vector for successful contour; None for unsuccessful contour (various
    cases apply)
    """
    # Get contour coordinates from contour
    
    #Contours with fewer than 5 points cannot be fit to ellipse - return None
    if contour.shape[0] < 5:
        return None
    (cx, cy), (short_axis, long_axis), angle = cv2.fitEllipse(contour)
    
    #contours without valid centroids cannot be processed - return None
    if math.isnan(cx) or math.isnan(cy):
        return None
    cx, cy = int(cx), int(cy)
    
    # Get a 64 x 64 center crop about each nucleus for GLCM features
    img_cell = get_cell_image(grayscale, cx, cy)
    img_cell_grey = np.pad(img_cell, [(0, 64 - img_cell.shape[0]), (0, 64 - img_cell.shape[1])],
                           mode='reflect')
    
    # 1. Generate contour features
    eccentricity = math.sqrt(1 - (short_axis / long_axis) ** 2)
    convex_hull = cv2.convexHull(contour)
    area, hull_area = cv2.contourArea(contour), cv2.contourArea(convex_hull)
    solidity = float(area) / hull_area
    arc_length = cv2.arcLength(contour, True)
    
    #it's possible in rare cases for the area to be evaluated as 0 - return None, avoid div-by-0 error
    if area == 0:
        return None
    roundness = (arc_length / (2 * math.pi)) / (math.sqrt(area / math.pi))
    intensity = get_mean_contour_intensity_grayscale(grayscale, contour)
    
    # 2. Generating GLCM features
    out_matrix = skimage.feature.greycomatrix(img_cell_grey, [1], [0])
    dissimilarity = skimage.feature.greycoprops(out_matrix, 'dissimilarity')[0][0]
    homogeneity = skimage.feature.greycoprops(out_matrix, 'homogeneity')[0][0]
    energy = skimage.feature.greycoprops(out_matrix, 'energy')[0][0]
    ASM = skimage.feature.greycoprops(out_matrix, 'ASM')[0][0]
    # Concatenate + Return all features
    x = [[short_axis, long_axis, angle, area, arc_length, eccentricity, roundness, solidity, intensity],
         [dissimilarity, homogeneity, energy, ASM]]
    return cx, cy, np.array(list(itertools.chain(*x)), dtype=np.float64)

def tile_level_feats(contours, image, x_coord, y_coord, tile_size=512):
    """
    Computes features for each nucleus in a tile and averages them to get feature 
    set for the entire tile. A future version of this function will incorporate
    neural-extracted features.
    
    :param contours: list of all contours (boundaries of nuclei) in a given tile.
    :param image: np array of tile, with dimensions (tile_dim_1, tile_dim_2, 3).
    :param x_coord: int for x_coord of upper-left-hand corner of tile.
    :param y_coord: int for y_coord of upper-left-hand corner of tile.
    :param tile_size: (optional) int for tile dimensions. Used for computing average
    centroid of tile. Default is 512.
    
    :return: feature vector as np array; integer count of number of nuclei in tile. 
    For the rare tile where no nuclei are (successfully) called from the segmentation
    mask, a dummy array will be returned with all entries -1, and a count of 0 nuclei. 
    This tile will then be removed in downstream processing and will not be included 
    in the final graph. Later versions of this function may elect to keep such tiles, 
    where neural features can make them more informative for graph analysis.
     
    """
    temp_data = []
    #Make grayscale image copy
    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    for contour in contours:
        output = get_basic_cell_features(image, grayscale, contour)
        if output is not None:
            cent_x, cent_y, features = output
            
            #re-type the data as a list - all lists will have identical len
            temp_data.append([cent_x, cent_y] + list(features))
    if len(temp_data) == 0:
        return np.array([-1 for i in range(15)]), 0
    
    #the non-ragged list can then be quickly averaged and converted to np array format
    result = np.average(np.asarray(temp_data), axis=0)
    
    #Compute the position for each node by averaging centroids of the tile's nuclei.
    result[0] = result[0]/512 + x_coord
    result[1] = result[1]/512 + y_coord
    return result, len(temp_data)

def KNN(G, num_neighbors=5, algorithm ='kmeans', branching=32, iterations=100, checks=16, **kwargs):
    """
    Condensed version of KNN which adds edges to graph.
    
    :param G: a networkx graph with nodes only.
    :param num_neighbors: (optional) int for max number of neighbors (edge) per node. Default 5
    :param algorithm: (optional) string for algorithm to be used. Default 'kmeans'
    :param branching: (optional) int; refer to FLANN docs. Default 32
    :param iterations: (optional) int; refer to FLANN docs. Default 100
    :param checks: (optional) int; refer to FLANN docs. Default 16
    
    :return: a networkx graph with edges added.
    """
    #Code directly transplanted from pathomic-fusion notebook 
    #First, build a "dataset"  using the centroid data - collect node data into a new array
    centroids = []
    for u, attrib in G.nodes(data=True):
        centroids.append(attrib['centroid'])

    cell_centroids = np.array(centroids).astype(np.float64)
    dataset = cell_centroids


    start = None

    #Run K-means
    for idx, attrib in tqdm(list(G.nodes(data=True))):
        start = idx

        #initialize the FLANN object 
        flann = FLANN()

        #??? - consider adding only one node at a time? 
        testset = np.array([attrib['centroid']]).astype(np.float64)

        #Calculate edges 
        results, dists = flann.nn(dataset, testset, num_neighbors=num_neighbors, algorithm=algorithm, 
                                  branching=branching, iterations=iterations, checks=checks)
        results, dists = results[0], dists[0]
        nns_fin = []
       # assert (results.shape[0] < 6)

        #Use results to draw in edges in the graph 
        for i in range(1, len(results)):
            G.add_edge(idx, results[i], weight = dists[i])
            nns_fin.append(results[i])

    return G

def from_networkx(G):
    r"""Converts a :obj:`networkx.Graph` or :obj:`networkx.DiGraph` to a
    :class:`torch_geometric.data.Data` instance.
    
    :param G: A networkx graph. (If this object was created by image-graph
    code, this graph will likely be undirected).
    
    :return: A pytorch geometric graph object (torch_geometric.data.Data).
    """

    G = G.to_directed() if not nx.is_directed(G) else G
    edge_index = torch.tensor(list(G.edges)).t().contiguous()

    keys = []
    keys += list(list(G.nodes(data=True))[0][1].keys())
    keys += list(list(G.edges(data=True))[0][2].keys())
    data = {key: [] for key in keys}

    for _, feat_dict in G.nodes(data=True):
        for key, value in feat_dict.items():
            data[key].append(value)
    for _, _, feat_dict in G.edges(data=True):
        for key, value in feat_dict.items():
            data[key].append(value)

    #Hopefully I can re-type the final dictionary value manually to avoid issues
    weights = data['weight']
    weights = [float(x) for x in weights]
    # weights = np.array(weights)
    data['weight'] = weights

    # THIS IS THE PROBLEMATIC PART
    for key in data.keys():
        data[key] = torch.tensor(data[key])
    #     print(key)

    data['edge_index'] = edge_index
    data = torch_geometric.data.Data.from_dict(data)
    data.num_nodes = G.number_of_nodes()

    return data

def to_networkx(data, node_attrs=None, edge_attrs=None, to_undirected=False,
                remove_self_loops=False):
    r"""Converts a :class:`torch_geometric.data.Data` instance to a
    :obj:`networkx.Graph` if :attr:`to_undirected` is set to :obj:`True`, or
    a directed :obj:`networkx.DiGraph` otherwise.

    Args:
        data (torch_geometric.data.Data): The data object.
        node_attrs (iterable of str, optional): The node attributes to be
            copied. (default: :obj:`None`)
        edge_attrs (iterable of str, optional): The edge attributes to be
            copied. (default: :obj:`None`)
        to_undirected (bool, optional): If set to :obj:`True`, will return a
            a :obj:`networkx.Graph` instead of a :obj:`networkx.DiGraph`. The
            undirected graph will correspond to the upper triangle of the
            corresponding adjacency matrix. (default: :obj:`False`)
        remove_self_loops (bool, optional): If set to :obj:`True`, will not
            include self loops in the resulting graph. (default: :obj:`False`)
            
    NOTE - this function is taken directly from pytorch geometric docs and
    has not been tested yet. 
    """

    if to_undirected:
        G = nx.Graph()
    else:
        G = nx.DiGraph()

    G.add_nodes_from(range(data.num_nodes))

    values = {}
    for key, item in data:
        if torch.is_tensor(item):
            values[key] = item.squeeze().tolist()
        else:
            values[key] = item
        if isinstance(values[key], (list, tuple)) and len(values[key]) == 1:
            values[key] = item[0]

    for i, (u, v) in enumerate(data.edge_index.t().tolist()):

        if to_undirected and v > u:
            continue

        if remove_self_loops and u == v:
            continue

        G.add_edge(u, v)
        for key in edge_attrs if edge_attrs is not None else []:
            G[u][v][key] = values[key][i]

    for key in node_attrs if node_attrs is not None else []:
        for i, feat_dict in G.nodes(data=True):
            feat_dict.update({key: values[key][i]})

    return G

def save_to_pytorch_geometric(G, save_path=None, **kwargs):
    """
    Code block for re-typing a pytorch geometric graph (tensor-ficating necessary attributes)
    and saving it.
    
    :param G: pytorch geometric graph object (torch_geometric.data.Data)
    
    :param save_path: (optional) string representing the path to which the graph should be saved.
    If not specified, the graph will be returned instead of saved. Default None.
    
    :return: pytorch geometric graph object (torch_geometric.data.Data), if save_path==None.
    """
    G = from_networkx(G)

    edge_attr_long = (G.weight.unsqueeze(1)).type(torch.LongTensor)
    G.edge_attr = edge_attr_long 

    edge_index_long = G['edge_index'].type(torch.LongTensor)
    G.edge_index = edge_index_long

    x_float = G['x'].type(torch.FloatTensor)
    G.x = x_float

    G['weight'] = None
    G['nn'] = None
    
    if save_path is None:
        return G
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(G, save_path)
    return None
    
def df2graph_basic(df, hovernet, device, save_name=None, save_masks=False, mask_save_name=None,
                   batch_size=10, num_workers=12, **kwargs):
    """
    An end-to-end function for building a Pytorch Geometric graph from WSI tiles with
    basic, nucleus-level features. 

    To use this function, ensure that a hovernet PathML model has been initialized
    and sent to the GPU with the above code block (GPU under variable name device, 
    hovernet under variable name hovernet) for proper inference.

    :param df: a Pandas DataFrame with one row per tile. This dataframe should have
    columns 'full_path' and 'slide_id' for identifying the originating WSI and 
    locating the tile image files and should contain tiles for ONLY one sample_id at
    a time. Additionally, columns 'x' and 'y' should be present for sorting the tiles
    in the dataframe (can be created easily from full_path name if necessary before 
    passing in the df).
    :param hovernet: pytorch HoverNet model for nucleus segmentation (instantiated by 
    make_model() function)
    :param device: GPU device for nucleus segmentation (instantiated by make_model() 
    function)
    :param save_name: (optional) string indicating a path/file name to which 
    the graph should be saved; filename should have extention '.pt'. Default None; 
    if not specified, the graph object will be returned.
    :param save_masks: (optional) bool indicating if segmentation masks should be
    saved. If True, a mask_save_name should ideally be specified as well. Default False.
    :param mask_save_name: (optional) str indicating a path/file name to which the 
    mask should be saved; should be specified if save_masks==True. If save_masks==True
    but mask_save_name==False, then the masks will be saved to the same location and
    filename as the graph, but with different extension. If save_masks==true, 
    mask_save_name==False, and save_name==None, then the masks will be returned as a 
    very large NumPy array.
    :param batch_size: (optional) int specifying batch size. Due to the intensity of
    HoverNet inference, larger batches may quickly cause memory errors. Default 10 
    (the empirical maximum for Tesla V100)
    :param num_workers: (optional) int specifying the number of parallel processes.
    Default 12

    :return: Pytorch geometric graph object, if no save_name is specified; large NumPy
    array of dimensions (num_tiles, tile_dim, tile_dim) of segmentation masks, if
    save_masks is True and both mask_save_name and save_name are None-type.
    """
    #Reorder the rows of dataframe by x,y coordinates. 
    try:
        df = df.reset_index()
    except ValueError:
        pass
    df = df.sort_values(by=['x', 'y'])
    df = df.reset_index()
    df = df.sort_values(by=['x', 'y'])
    df = df.reindex([i for i in range(df.shape[0])])
    try:
        df = df.drop(labels=['index'], axis=1)
    except KeyError:
        pass
    try:
        df = df.drop(labels=['level_0'], axis=1)
    except KeyError:
        pass

#     #Extract the slide_id string 
#     slide_id = None
#     try:
#         slide_id  df.slide_id.to_numpy()[0]
#     except AttributeError:
#         slide_id = df.index.to_numpy()[0]

    #Get an array of path names
    paths = df.full_path.to_numpy()
    print('Number of tiles:', paths.shape[0])

    #Get matching array of image arrays; rearrange axes for torch inference
    tiles = np.array([as_array(path) for path in paths])
    tiles = np.moveaxis(tiles, 3, 1)

    #Build simple dataloader
    tile_data = torch.utils.data.DataLoader(tiles, batch_size=batch_size, num_workers=num_workers)
    #Paths are strings which PyTorch HATES - so they live in a temporary array.
    paths_temp = None

    #Initialize empty objects
    G = nx.Graph()
    if save_masks:
        mask_arr = None

    node_counter = 0
    #pass tiles to the GPU for segmentation inference in small batches
    with torch.no_grad():
        for batch, data in tqdm(enumerate(tile_data)):
            # send the data to the GPU
            paths_temp = paths[batch_size*batch:batch_size*(batch+1)]
            images = data.float().to(device)

            # pass thru network to get predictions
            outputs = hovernet(images)
            _, preds_classification = post_process_batch_hovernet(outputs, n_classes=6)

            #get images, masks ready for feature extraction
            images = np.moveaxis(data.numpy(), 1, 3).astype('uint8')
            masks = np.sum(preds_classification, axis=1)
            masks = np.where(masks > 0, 255, 0)
            masks = masks.astype('uint8')
            
            if save_masks and batch == 0:
                mask_arr = masks
            elif save_masks:
                mask_arr = np.concatenate((mask_arr, masks))
                
                

            #feature extraction
            for path, image, mask in zip(paths_temp, images, masks):
                    #Get tile coordinates 
                    x_coord = int(path.split('/')[-1].split('_')[0])
                    y_coord = path.split('/')[-1].split('_')[1]
                    y_coord = int(y_coord[:y_coord.find('.')])

                    #Get contours for the nuclei in the tile
                    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                    #Extract and average the nucleus-level features to get tile-level features
                    features, num_nuclei = tile_level_feats(contours, image, x_coord, y_coord)

                    #Add features to graph
                    if num_nuclei > 0:
                        G.add_node(node_counter, centroid=np.array([features[0], features[1]], dtype=np.float32), 
                                  x=np.array([num_nuclei] + list(features[2:])).astype(np.float32))
                        node_counter += 1

    #Add edges to graph and save it
    G = KNN(G)
    G = save_to_pytorch_geometric(G, save_path=save_name)
    
    #save array if necessary
    #Note - either array and graph will be returned; or just graph returned; or nothing at all
    #Array will never be returned alone; it will be saved whenever possible (to try and save RAM)
    return_masks = False
    if save_masks:
        if mask_save_name is not None:
            os.makedirs(os.path.dirname(mask_save_name), exist_ok=True)
            np.save(mask_save_name, mask_arr)
        elif mask_save_name is None and save_name is not None:
            np.save(save_name.replace('.pt', '.npy'), mask_arr)
        else:
            return_masks = True

    #Manual clearing of the bulkiest variables - save memory if multiple slides are to be run
    del paths
    del paths_temp
    del tiles
    del tile_data
    del images 
    del outputs
    if not save_masks:
        del mask_arr
    del _
    del preds_classification
    del masks
    del contours

    if return_masks and G is not None:
        return G, mask_arr
    elif G is not None:
        return G
    
