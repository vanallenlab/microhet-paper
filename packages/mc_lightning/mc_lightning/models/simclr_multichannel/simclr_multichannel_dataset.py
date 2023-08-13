import torch
import torch.utils.data as data
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from torchvision import transforms

from PIL import Image
from mc_lightning.utilities.utilities import pil_loader


# loads saved .pth arrays, transforms twice
class SlideDataset(data.Dataset):
    def __init__(self, paths, slide_ids, labels, transform_compose):
        self.paths = paths
        self.slide_ids = slide_ids
        self.labels = labels
        self.transform = transform_compose

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img_path = self.paths[index]
        # tensor_file = torch.from_numpy(torch.load(img_path)).permute(2, 0, 1)
        tensor_file = torch.from_numpy(torch.load(img_path)).permute(2, 0, 1).float()
        # print(tensor_file.shape)
        transformed_tensors = self.transform(tensor_file)
        slide_id = self.slide_ids[index]
        label = self.labels[index]

        # model expects `(img1, img2), y = batch` at `self.shared_step` call
        return transformed_tensors, (label, slide_id)


