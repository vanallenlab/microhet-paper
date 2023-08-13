import torch
import torch.utils.data as data
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from tqdm import tqdm 
from PIL import Image
from mc_lightning.utilities.utilities import pil_loader

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import numpy as np
from skimage import color

# import staintools
# from p_tqdm import p_umap
import warnings

class SlideDataset(data.Dataset):
    """
    Modification of vanilla `tmb_bot.utilities.Dataset` class to facilitate having
    a label for classification as well as the slide name itself
    """
    def __init__(self, paths, slide_ids, labels, transform_compose, bw = 'None'):
        """
        Paths and labels should be array like
        """
        self.paths = paths
        self.slide_ids = slide_ids
        self.labels = labels
        self.transform = transform_compose
        self.bw = bw

    def __len__(self):
        'Denotes the total number of samples'
        return self.paths.shape[0]

    def __getitem__(self, index):
        'Generates one sample of data'

        img_path = self.paths[index]
        pil_file = pil_loader(img_path, self.bw)
        pil_file = self.transform(pil_file)
        slide_id = self.slide_ids[index]
        label = self.labels[index]

        return pil_file, label, slide_id


class SlideDataset_amenable_to_batch_label_models(data.Dataset):
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

        return pil_file, (label, label), slide_id

class BESTDataset(data.Dataset): #Batch Effect Created Using Stain Tools
    """
    Modification of vanilla `tmb_bot.utilities.Dataset` class to facilitate having
    a label for classification as well as the slide name itself
    """
    def __init__(self, paths, slide_ids, labels, transform_compose, stain_batch = None, augment = None, p = None):
        """
        Paths and labels should be array like
        """

        # print('initing BEST Dataset')

        self.paths = paths
        self.slide_ids = slide_ids
        self.labels = labels
        self.transform = transform_compose
        self.stain_batch = stain_batch
        self.augment = augment
        self.p = p

        temp1_path = "/home/jupyter/LUAD/Lung/be template 1.png"
        temp2_path = "/home/jupyter/LUAD/Lung/be template 2.png"
        self.temp1 = staintools.read_image(temp1_path)
        self.temp2 = staintools.read_image(temp2_path)
        self.temp1 = staintools.LuminosityStandardizer.standardize(self.temp1)
        self.temp2 = staintools.LuminosityStandardizer.standardize(self.temp2)

        self.normalizer1 = staintools.StainNormalizer(method='macenko')
        self.normalizer2 = staintools.StainNormalizer(method='macenko')

        self.normalizer1.fit(self.temp1)
        self.normalizer2.fit(self.temp2)

        # data = []
        # for i, path in enumerate(self.paths):
        #     to_transform = staintools.read_image(path)
        #     to_transform = staintools.LuminosityStandardizer.standardize(to_transform)
        #     data.append([self.normalizer1.transform(to_transform), self.normalizer2.transform(to_transform)])

    def __len__(self):
        'Denotes the total number of samples'
        return self.paths.shape[0]

    def __getitem__(self, index):
        'Generates one sample of data'

        img_path = self.paths[index]
        if self.augment == 'staintools':
            to_transform = staintools.read_image(img_path)
            # Standardize brightness (optional, can improve the tissue mask calculation)
            to_transform = staintools.LuminosityStandardizer.standardize(to_transform)
            try:
                dice = np.random.rand() 
                p = self.p

                if self.stain_batch == '1':                    
                    if dice > p:
                        transformed = self.normalizer1.transform(to_transform)
                        stain_label = 0
                    else:
                        transformed = self.normalizer2.transform(to_transform)
                        stain_label = 1
                    
                elif self.stain_batch == '2':                                        
                    if dice > 1 - p:
                        transformed = self.normalizer1.transform(to_transform)
                        stain_label = 0
                    else:
                        transformed = self.normalizer2.transform(to_transform)    
                        stain_label = 1

                pil_file = Image.fromarray(transformed)

            except Exception as e:
                print(e)
                stain_label = 0
                pil_file = Image.fromarray(to_transform)

            # print(8)
            # print('transformed')
            
            # print(9)

            pil_file = self.transform(pil_file)
            # print(10)
            slide_id = self.slide_ids[index]
            label = self.labels[index]

            return pil_file, (label, stain_label), slide_id

        else:
            pil_file = pil_loader(img_path)
            # print('augment not stain tools')

        pil_file = self.transform(pil_file)
        # print(10)
        slide_id = self.slide_ids[index]
        label = self.labels[index]

        return pil_file, (label, label), slide_id

class BEST_preprocessed_Dataset(data.Dataset): #Batch Effect Created Using Stain Tools
    """
    Modification of vanilla `tmb_bot.utilities.Dataset` class to facilitate having
    a label for classification as well as the slide name itself
    """
    def __init__(self, paths, slide_ids, labels, transform_compose, stain_batch = None, augment = None, p = None):
        """
        Paths and labels should be array like
        """

        # print('initing BEST Dataset')

        self.paths = paths
        self.slide_ids = slide_ids
        self.labels = labels
        self.transform = transform_compose
        self.stain_batch = stain_batch
        self.augment = augment
        self.p = p

        temp1_path = "/home/jupyter/LUAD/Lung/be template 1.png"
        temp2_path = "/home/jupyter/LUAD/Lung/be template 2.png"
        self.temp1 = staintools.read_image(temp1_path)
        self.temp2 = staintools.read_image(temp2_path)
        self.temp1 = staintools.LuminosityStandardizer.standardize(self.temp1)
        self.temp2 = staintools.LuminosityStandardizer.standardize(self.temp2)

        self.normalizer1 = staintools.StainNormalizer(method='macenko')
        self.normalizer2 = staintools.StainNormalizer(method='macenko')

        self.normalizer1.fit(self.temp1)
        self.normalizer2.fit(self.temp2)

            
        # try:
        #     path = self.paths[0]
        #     img = path.split('/')[-1].split('.')[0]
        #     pil_file1 = pil_loader(path[:-len(img + '.png')] + img + '_t1' + '.png')
        #     pil_file2 = pil_loader(path[:-len(img + '.png')] + img + '_t2' + '.png')
        # except Exception as e:
        #     print(e)
        #     p_umap(f, self.paths, num_cpus = 12)
        
        #UnComment if planning to load some new data
        # from imports import stain_transform_images
        # p_umap(stain_transform_images, self.paths, num_cpus = 12)
        
                
    def __len__(self):
        'Denotes the total number of samples'
        return self.paths.shape[0]

    def __getitem__(self, index):
        'Generates one sample of data'

        img_path = self.paths[index]
        img = img_path.split('/')[-1].split('.')[0]
        pil_file1 = pil_loader(img_path[:-len(img + '.png')] + img + '_t1' + '.png')
        pil_file2 = pil_loader(img_path[:-len(img + '.png')] + img + '_t2' + '.png')

        if self.augment == 'staintools':
            # to_transform = staintools.read_image(img_path)
            # Standardize brightness (optional, can improve the tissue mask calculation)
            # to_transform = staintools.LuminosityStandardizer.standardize(to_transform)
            
            dice = np.random.rand() 
            p = self.p
            stain_batch = self.stain_batch[index]

            if stain_batch == 0:                    
                if dice < p:
                    stain_label = 0                        
                else:
                    stain_label = 1
                
            elif stain_batch == 1:                                        
                if dice < 1 - p:
                    stain_label = 0
                else:
                    stain_label = 1

            pil_file = [pil_file1, pil_file2][stain_label]
                        
            pil_file = self.transform(pil_file)
            # print(10)
            slide_id = self.slide_ids[index]
            label = self.labels[index]

            return pil_file, (label, stain_label), slide_id

        else:
            pil_file = pil_loader(img_path)
            print('augment not stain tools')

        pil_file = self.transform(pil_file)
        slide_id = self.slide_ids[index]
        label = self.labels[index]

        return pil_file, (label, label), slide_id

class CombinedDataset(data.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        # print(self.datasets)
    
    def __len__(self):
        'Denotes the total number of samples'
        return sum([i.paths.shape[0] for i in self.datasets])

    def __getitem__(self, index):
        'Generates one sample of data'
        #Get a dataset
        which_dataset = int(index >= self.datasets[0].__len__())  #random.choice(list(range(len(self.datasets))))
        
        #Index into it
        start_pos = sum([self.datasets[i].__len__() for i in range(which_dataset)])

        residual_index = index - start_pos

        # print(50 * '*')
        # print('index', index)
        # print('which_dataset', which_dataset)
        # print('which_dataset len', self.datasets[which_dataset].__len__())
        # print('start_pos', start_pos)
        # print('residual index', residual_index)

        #Get attrs of interest
        pil_file, label, slide_id = self.datasets[which_dataset].__getitem__(residual_index)
        
        # img_path = self.paths[index]
        # pil_file = pil_loader(img_path)
        # pil_file = self.transform(pil_file)
        # slide_id = self.slide_ids[index]
        # label = self.labels[index]

        return pil_file, label, slide_id

class SlideDataset_hsv(data.Dataset):
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
        # pil_file = pil_file.convert(mode = 'HSV')
        pil_file = self.transform(pil_file)        
        slide_id = self.slide_ids[index]
        label = self.labels[index]

        return pil_file, label, slide_id

class SiameseSlideDataset(data.Dataset):
    """
    Modification of vanilla `tmb_bot.utilities.Dataset` class to facilitate having
    a label for classification as well as the slide name itself
    """
    def __init__(self, paths1, paths2, slide_ids1, slide_ids2, labels, transform_compose):
        """
        Paths and labels should be array like
        """
        self.paths1 = paths1
        self.paths2 = paths2

        self.slide_ids1 = slide_ids1
        self.slide_ids2 = slide_ids2

        self.labels = labels
        
        self.transform = transform_compose

    def __len__(self):
        'Denotes the total number of samples'
        return self.paths1.shape[0]

    def __getitem__(self, index):
        'Generates one sample of data'

        img_path1 = self.paths1[index]
        img_path2 = self.paths2[index]

        pil_file1 = pil_loader(img_path1)
        pil_file2 = pil_loader(img_path2)

        pil_file1 = self.transform(pil_file1)
        pil_file2 = self.transform(pil_file2)

        slide_id1 = self.slide_ids1[index]
        slide_id2 = self.slide_ids2[index]

        label = self.labels[index]

        return pil_file1, pil_file2, label, slide_id1, slide_id2

class WSIV(data.Dataset):
    """
    Modification of vanilla `tmb_bot.utilities.Dataset` class to facilitate having
    a label for classification as well as the slide name itself
    """
    def __init__(self, paths, slide_ids, labels, transform_compose, 
    num_ettes = 2, wsi_folder = '/home/jupyter/LUAD/wsi_images/', non_lin = None, how = 'index'):
        """
        Paths and labels should be array like
        """
        self.paths = paths
        self.slide_ids = slide_ids
        self.labels = labels
        self.transform = transform_compose
        self.num_ettes = num_ettes
        self.embed_3d = np.random.rand(self.num_ettes, 2048, 3)        
        self.non_lin = non_lin
        self.wsi_folder = wsi_folder
        self.how = how
        self.indexes = np.random.randint(low = 0, high = 2048, size = (self.num_ettes, 3)) 

    def __len__(self):
        'Denotes the total number of samples'
        return self.paths.shape[0]

    def __getitem__(self, index):
        'Generates one sample of data'
        img_path = self.paths[index]
        wsi = np.load(self.wsi_folder + img_path)

        # pil_file = pil_loader(img_path)
        if self.how == 'transform':
            wsiv = wsi @ self.embed_3d[index%self.num_ettes]
        elif self.how == 'index':
            wsiv = wsi[:, :, self.indexes[index%self.num_ettes]]
            wsiv = np.ascontiguousarray(wsiv)
        
        if self.non_lin != None:
            wsiv = self.non_lin(wsiv)

        wsiv.resize((512, 512, 3))
        wsiv = torch.from_numpy(wsiv)
        wsiv = wsiv.permute(2, 0, 1).float()
        # wsiv = self.transform(wsiv)
        
        slide_id = self.slide_ids[index]
        label = self.labels[index]

        return wsiv, label, slide_id
