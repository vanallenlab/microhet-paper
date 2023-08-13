import numpy as np
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
from torch.optim.lr_scheduler import StepLR
import albumentations as

from pathml.datasets.pannuke import PanNukeDataModule
from pathml.ml.hovernet import HoVerNet, loss_hovernet, post_process_batch_hovernet, _HoverNetDecoder
from pathml.ml.utils import wrap_transform_multichannel, dice_score
from pathml.utils import plot_segmentation

from PIL import Image

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


nonsubset = paths.drop(list(subset.index.unique()))
nonsubset_pred_tumor = nonsubset.loc[nonsubset.prob_tumor >= 0.7].sample(150)
pred_tumor_dataset = SlideDataset(
    paths=nonsubset_pred_tumor.full_path.values,
    slide_ids=nonsubset_pred_tumor.index.values,
    labels=nonsubset_pred_tumor.prob_tumor.values,
    transform_compose=RGBEvalTransform(512, 256, norm_mean=[0, 0, 0], norm_std=[1, 1, 1])
)

pred_tumor_dataloader = DataLoader(pred_tumor_dataset, batch_size=BATCH_SIZE, num_workers=WORKERS, shuffle=True)

preds_detection_store = []
ft_preds_classification_store = []
labels_store = []
ids_store = []
img_store = []


# load the best model
checkpoint = torch.load("/mnt/disks/image_data/kidney/hover/hovernet_pannuke.pt", map_location='cpu')

n_classes_pannuke = 6

# load the model
hovernet = HoVerNet(n_classes=n_classes_pannuke)
hovernet = torch.nn.DataParallel(hovernet)
hovernet.load_state_dict(checkpoint)

hovernet = hovernet.module

hovernet.to('cuda');

HEAD_ONLY = True
if HEAD_ONLY:
    print('only tuning a FC head layer')

# ALTERNATIVE: start with pretrained NC
if not HEAD_ONLY:
    nc_branch_pretrain_clone = copy.deepcopy(hovernet.nc_branch)
    new_decoder = nc_branch_pretrain_clone
    new_decoder.to('cuda')

# create new head layer to push NC to 3 classes
decoder_post_fc = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1)
decoder_post_fc.to('cuda');



for batch_idx, (imgs, labels, ids) in enumerate(pred_tumor_dataloader):
    ## use pretrained model to get initial outputs to work with
    with torch.no_grad():
        encoding = hovernet.encoder(imgs.cuda())
        frozen_outputs = hovernet(imgs.cuda())  # TODO modify forward pass to give encoding and outputs?
        froz_np, froz_hv, froz_nc = frozen_outputs

        if HEAD_ONLY:
            nc_branch_out = hovernet.nc_branch(encoding)  # decode

        # get baseline nuclear segmentations
        preds_detection, preds_classification = post_process_batch_hovernet(frozen_outputs, n_classes=6)
        # generate pseudomasks
        pseudomasks = get_pseudomask(preds_detection, labels)

        if not HEAD_ONLY:
            nc_branch_out = new_decoder(encoding)  # decode
        ft_nc = decoder_post_fc(nc_branch_out)  # transform

        updated_outputs = [froz_np, froz_hv, ft_nc]

        preds_detection_store.append(preds_detection)
        labels_store.append(labels)
        ids_store.append(ids)
        img_store.append(imgs.cpu())

    ft_preds_detection, ft_preds_classification = post_process_batch_hovernet(updated_outputs, n_classes=3)
    ft_preds_classification_store.append(ft_preds_classification)

preds_detection_store = np.concatenate(preds_detection_store)
ft_preds_classification_store = np.concatenate(ft_preds_classification_store)
labels_store = np.concatenate(labels_store)
ids_store = np.concatenate(ids_store)
img_store = np.concatenate(img_store)

store_agg = {
    'preds_detection_store': preds_detection_store,
    'ft_preds_classification_store': ft_preds_classification_store,
    'labels_store': labels_store,
    'ids_store': ids_store,
    'img_store': img_store,
}

torch.save(store_agg, '20210212_ptvnt_geq070_seg_class_preds_finetuned_head_only.pth')



###### nucleus characterization
feat_store = []
coord_store = []
label_exp_store = []
contour_store = []
tile_rel_idx = []
contour_rel_idx = []

for idx in range(len(preds_detection_store)):
    binary = ((preds_detection_store[idx] > 0) * 255).astype(np.uint8)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_store.append(contours)

    temp_img = (img_store[idx] * 255).astype(np.uint8).transpose((1, 2, 0))
    if len(contours) > 0:
        for contour_idx, contour in enumerate(contours):
            try:
                #                 feats, cx, cy = get_cell_features_mod(temp_img, contour, resnet_encoder)
                feats, cx, cy = get_cell_encoding(temp_img, contour, resnet_encoder)
                feat_store.append(feats)
                coord_store.append((cx, cy))
                label_exp_store.append(ids_store[idx])
                tile_rel_idx.append(idx)
                contour_rel_idx.append(contour_idx)
            except:
                pass