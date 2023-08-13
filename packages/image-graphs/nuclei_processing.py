
### based directly on pathomic fusion's cell graph construction code
##### https://github.com/mahmoodlab/PathomicFusion/blob/master/notebooks/cell_graph.ipynb
import cv2
import skimage
import os
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import itertools
import math
import numpy as np
import pandas as pd


from pathml.ml.hovernet import HoVerNet, loss_hovernet, post_process_batch_hovernet, _HoverNetDecoder

to_pil = transforms.ToPILImage()

def get_cell_image(img, cx, cy, size=512):
    cx = 32 if cx < 32 else size - 32 if cx > size - 32 else cx
    cy = 32 if cy < 32 else size - 32 if cy > size - 32 else cy
    return img[cy - 32:cy + 32, cx - 32:cx + 32, :]


def get_encoder_features(cell, encoder):
    if torch.cuda.is_available():
        cell = torch.Tensor(cell).permute(2, 0, 1).unsqueeze(0)
        feats = encoder(cell.cuda()).flatten(1).squeeze(0)
        return feats.cpu().detach().numpy()
    else:
        cell = torch.Tensor(cell).permute(2, 0, 1).unsqueeze(0)
        feats = encoder(cell).flatten(1).squeeze(0)
        return feats.detach().numpy()



def seg2graph(img, contours):
    G = nx.Graph()

    contours = [c for c in contours if c.shape[0] > 5]

    for v, contour in enumerate(contours):
        features, cx, cy = get_cell_features(img, contour)
        G.add_node(v, centroid=[cx, cy], x=features)

    if v < 5: return None
    return G



def get_contour_features(contour):
    # Get contour coordinates from contour
    try:
        (cx, cy), (short_axis, long_axis), angle = cv2.fitEllipse(contour)
        cx, cy = int(cx), int(cy)

        eccentricity = math.sqrt(1 - (short_axis / long_axis) ** 2)
        convex_hull = cv2.convexHull(contour)
        area, hull_area = cv2.contourArea(contour), cv2.contourArea(convex_hull)
        solidity = float(area) / hull_area
        arc_length = cv2.arcLength(contour, True)
        roundness = (arc_length / (2 * math.pi)) / (math.sqrt(area / math.pi))
    except:
        area = np.nan
        roundness = np.nan
        eccentricity = np.nan
        solidity = np.nan
        cx = np.nan
        cy = np.nan
    return cx, cy, eccentricity, area, roundness, solidity


def get_cell_features_generic_encoder(img, contour, encoder):
    """
    Modified generic version that allows for any pytorch encoder to be passed

    :param img: tensor image
    :param contour: contour produced from nucleus segmentation
    :param encoder: frozen pretrained encoder object (ex resnet50 pre-classifier)
    :return: concatenated feature vector and (x,y) coordinate of segmentation centroid
    """
    # Get contour coordinates from contour
    (cx, cy), (short_axis, long_axis), angle = cv2.fitEllipse(contour)
    cx, cy = int(cx), int(cy)

    # Get a 64 x 64 center crop over each cell
    img_cell = get_cell_image(img, cx, cy)

    grey_region = cv2.cvtColor(img_cell, cv2.COLOR_RGB2GRAY)
    img_cell_grey = np.pad(grey_region, [(0, 64 - grey_region.shape[0]), (0, 64 - grey_region.shape[1])],
                           mode='reflect')

    # 1. Generating contour features
    eccentricity = math.sqrt(1 - (short_axis / long_axis) ** 2)
    convex_hull = cv2.convexHull(contour)
    area, hull_area = cv2.contourArea(contour), cv2.contourArea(convex_hull)
    solidity = float(area) / hull_area
    arc_length = cv2.arcLength(contour, True)
    roundness = (arc_length / (2 * math.pi)) / (math.sqrt(area / math.pi))

    # 2. Generating GLCM features
    out_matrix = skimage.feature.greycomatrix(img_cell_grey, [1], [0])
    dissimilarity = skimage.feature.greycoprops(out_matrix, 'dissimilarity')[0][0]
    homogeneity = skimage.feature.greycoprops(out_matrix, 'homogeneity')[0][0]
    energy = skimage.feature.greycoprops(out_matrix, 'energy')[0][0]
    ASM = skimage.feature.greycoprops(out_matrix, 'ASM')[0][0]

    # 3. Generating encoder features
    encoder_feats = get_encoder_features(img_cell, encoder)

    # Concatenate + Return all features
    x = [[short_axis, long_axis, angle, area, arc_length, eccentricity, roundness, solidity],
         [dissimilarity, homogeneity, energy, ASM],
         encoder_feats]

    return np.array(list(itertools.chain(*x)), dtype=np.float64), cx, cy


def get_cell_encoding(img, contour, encoder):
    # Get contour coordinates from contour
    (cx, cy), (short_axis, long_axis), angle = cv2.fitEllipse(contour)
    cx, cy = int(cx), int(cy)

    # Get a 64 x 64 center crop over each cell
    img_cell = get_cell_image(img, cx, cy)

    grey_region = cv2.cvtColor(img_cell, cv2.COLOR_RGB2GRAY)
    img_cell_grey = np.pad(grey_region, [(0, 64 - grey_region.shape[0]), (0, 64 - grey_region.shape[1])],
                           mode='reflect')

    resnet_feats = get_resnet_features(img_cell, encoder)

    return resnet_feats, cx, cy


def get_mean_contour_intensity_grayscale(gray_img, contours):
    """
    Proxy for how dark/light the pixels are within the segmented nucleus area

    :param gray_img: cv2 converted grayscale image of segmented nucleus area
    :param contours: contour produced from nucleus segmentation
    :return: mean pixel intensity over nucleus contour area
    """
    img_size = gray_img.shape[0]
    assert gray_img.shape[0] == gray_img.shape[1]
    
    nucl_mask = np.zeros((img_size, img_size))
    cv2.fillPoly(nucl_mask, pts=contours, color=(1.));  # use contour area to make a mask
    z = np.ma.masked_array(data=gray_img,
                           mask=(nucl_mask != 1.))  # mask masked version to select out only the contour area

    return z.compressed().mean()  # take mean grayscale pixel intensity over contour area


def get_contour_features(contour, gray_img):
    """
    :param contour: contour produced from nucleus segmentation
    :param gray_img: cv2 converted grayscale image of segmented nucleus area
    :return: (x,y) coordinates, contour features of nucleus
    """
    # Get contour coordinates from contour
    try:
        (cx, cy), (short_axis, long_axis), angle = cv2.fitEllipse(contour)
        cx, cy = int(cx), int(cy)

        eccentricity = math.sqrt(1 - (short_axis / long_axis) ** 2)
        convex_hull = cv2.convexHull(contour)
        area, hull_area = cv2.contourArea(contour), cv2.contourArea(convex_hull)
        solidity = float(area) / hull_area
        arc_length = cv2.arcLength(contour, True)
        roundness = (arc_length / (2 * math.pi)) / (math.sqrt(area / math.pi))

        # my addition: mean pixel intensity of grayscale img in contour area
        intensity = get_mean_contour_intensity_grayscale(gray_img, [contour])


    except Exception as e:
#         print(e)
        area = np.nan
        roundness = np.nan
        eccentricity = np.nan
        solidity = np.nan
        cx = np.nan
        cy = np.nan
        intensity = np.nan
    return cx, cy, eccentricity, area, roundness, solidity, intensity


def run_contour_calling_three_class(mask, img, file_id, prob_tumor):
    """
    Assuming setup where `mask` is the output of hovernet classification where
    the class setup was as follows:
        0: "Background"
        1: "Tumor region"
        2: "non-tumor region"
    """
    example_store = []
    contour_store = []
    for channel_idx in range(3):
        binary = ((mask[channel_idx] > 0) * 255).astype(np.uint8)
        gray_img = cv2.cvtColor(img.transpose((1, 2, 0)), cv2.COLOR_RGB2GRAY)

        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            contour_fts = [get_contour_features(x, gray_img) for x in contours]

            temp_df = pd.DataFrame(contour_fts)
            temp_df.columns = ['cx', 'cy', 'eccentricity', 'area', 'roundness', 'solidity', 'intensity']
            temp_df['file_id'] = file_id
            temp_df['prob_tumor'] = prob_tumor
            temp_df['nc_bin'] = channel_idx
            temp_df['nc_category'] = temp_df.nc_bin.map({0: 'background', 1: 'tumor', 2: 'stroma'})
            example_store.append(temp_df)
            contour_store.extend(contours)

    example_store = pd.concat(example_store).reset_index(drop=True)
    example_store['contour_idx'] = range(len(example_store))

    return example_store, contour_store


def run_contour_calling_binary(mask, img, file_id, prob_tumor):
    """
    Assuming setup where `mask` is the general nucleus segmentation calls of hovernet (NP branch)
    """
    example_store = []
    contour_store = []
    binary = ((mask > 0) * 255).astype(np.uint8)
    gray_img = cv2.cvtColor(img.transpose((1, 2, 0)), cv2.COLOR_RGB2GRAY)

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        contour_fts = [get_contour_features(x, gray_img) for x in contours]

        temp_df = pd.DataFrame(contour_fts)
        temp_df.columns = ['cx', 'cy', 'eccentricity', 'area', 'roundness', 'solidity', 'intensity']
        temp_df['file_id'] = file_id
        temp_df['prob_tumor'] = prob_tumor

        example_store.append(temp_df)
        contour_store.extend(contours)

    example_store = pd.concat(example_store).reset_index(drop=True)
    example_store['contour_idx'] = range(len(example_store))

    return example_store, contour_store


def run_til_tumor_stroma_filtering(contour_df, contours, intensity_cutoff=76.5, roundness_cutoff=1.125,
                                   area_cutoff=110, crop_size=512):
    """
    Assuming setup where contours were called based on the output of hovernet classification where
    the class setup was as follows:
        0: "Background"
        1: "Tumor region"
        2: "non-tumor region"

    Stores categorical assignment as {'til','tumor','stroma'}
    Returns RGB image for visualizing these categorical assignments wrt segmentation masks
    """
    im = np.zeros((crop_size, crop_size, 3))

    crit0 = (contour_df.area < area_cutoff) & (contour_df.intensity < intensity_cutoff) & (
    contour_df.roundness < roundness_cutoff)
    crit0 = crit0 & (contour_df.nc_bin != 2)
    contour_df.loc[crit0, 'tissue_class'] = 'til'
    for contour_idx in np.where(crit0)[0]:
        cv2.fillPoly(im, pts=[contours[contour_idx]], color=(150, 0, 0))

    crit1 = ~crit0 & (contour_df.nc_bin != 2)
    contour_df.loc[crit1, 'tissue_class'] = 'tumor'
    for contour_idx in np.where(crit1)[0]:
        cv2.fillPoly(im, pts=[contours[contour_idx]], color=(0, 150, 0))

    crit2 = contour_df.nc_bin == 2
    contour_df.loc[crit2, 'tissue_class'] = 'stroma'
    for contour_idx in np.where(crit2)[0]:
        cv2.fillPoly(im, pts=[contours[contour_idx]], color=(0, 0, 150))

    return im


def process_slide(single_slide_dataloader, hovernet, out_dir = './hovernet_inference_outputs'):
    df_agg = []
    contour_agg = []
    with torch.no_grad():
        hovernet.eval()
        for batch_idx, (imgs, labels, ids) in enumerate(single_slide_dataloader):
            imgs = imgs * 255.  # convention for this model code
            out = hovernet(imgs.cuda())
            pred_detection, pred_classification = post_process_batch_hovernet(out, 3)

            for rel_idx, (img, label, file_id) in enumerate(zip(imgs, labels, ids)):
                try:
                    tx, ty = label
                    contour_info, contours = run_contour_calling_three_class(
                        mask=pred_classification[rel_idx],
                        img=img.numpy(),
                        file_id=file_id,
                        prob_tumor='XXXXX', # TODO consider refactoring to not use prob_tumor at all
                    )

                    vis_im = run_til_tumor_stroma_filtering(contour_info, contours)

                    # assign tile x,y label as well
                    contour_info['tx'] = tx.item()
                    contour_info['ty'] = ty.item()
                    
                    # save detections + classifications
                    #torch.save(vis_im, os.path.join(out_dir, f'hovernet_finetuned_preds_{file_id}_{tx.item()}_{ty.item()}.pt'))
                    out_img = to_pil(torch.Tensor(vis_im).permute(2,0,1)/150)
                    out_img.save(os.path.join(out_dir, f'hovernet_finetuned_preds_{file_id}_{tx.item()}_{ty.item()}.png'))

                    df_agg.append(contour_info)
                    #contour_agg.extend(contours)
                except Exception as e:
                    print(f'Issue with batch {batch_idx}, file {rel_idx}')
                    print(e)

    df_agg = pd.concat(df_agg)

    return df_agg

# to_tensor = transforms.ToTensor()
# image_modules = list(models.resnet50(pretrained=True, progress=False).children())[:-1]
# resnet_encoder = nn.Sequential(*image_modules)
# resnet_encoder.eval();
# if torch.cuda.is_available():
#     resnet_encoder.cuda();




# ###### nucleus characterization
# df_store = []
# # for idx in range(2):
# for idx in range(len(store_agg['preds_detection_store'])):
#     binary = ((store_agg['preds_detection_store'][idx] > 0) * 255).astype(np.uint8)
#     img = store_agg['img_store'][idx]
#     gray_img = cv2.cvtColor(img.transpose((1, 2, 0)), cv2.COLOR_RGB2GRAY)

#     contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     if len(contours) > 0:
#         contour_fts = [get_contour_features(x, gray_img) for x in contours]

#         temp_df = pd.DataFrame(contour_fts)
#         temp_df.columns = ['cx', 'cy', 'eccentricity', 'area', 'roundness', 'solidity', 'intensity']
#         temp_df['file_id'] = store_agg['ids_store'][idx]
#         temp_df['prob_tumor'] = store_agg['labels_store'][idx]
#         df_store.append(temp_df)

# nuc_info = pd.concat(df_store)


# feat_store = []
# coord_store = []
# label_exp_store = []
# contour_store = []
# tile_rel_idx = []
# contour_rel_idx = []
#
# for idx in range(len(preds_detection_store)):
#     binary = ((preds_detection_store[idx] > 0) * 255).astype(np.uint8)
#     contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     contour_store.append(contours)
#
#     temp_img = (img_store[idx] * 255).astype(np.uint8).transpose((1, 2, 0))
#     if len(contours) > 0:
#         for contour_idx, contour in enumerate(contours):
#             try:
#                 #                 feats, cx, cy = get_cell_features_mod(temp_img, contour, resnet_encoder)
#                 feats, cx, cy = get_cell_encoding(temp_img, contour, resnet_encoder)
#                 feat_store.append(feats)
#                 coord_store.append((cx, cy))
#                 label_exp_store.append(ids_store[idx])
#                 tile_rel_idx.append(idx)
#                 contour_rel_idx.append(contour_idx)
#             except:
#                 pass




# INTENSITY_CUTOFF = 0.3
# ROUNDNESS_CUTOFF = 1.125
# AREA_CUTOFF = 110
#
#
# set_rc(20,8)
# # for idx in np.random.randint(0, len(preds_detection_store), 10):
# for idx in range(50,60):
#
#     print(f'intensity cutoff: {INTENSITY_CUTOFF:.2f}')
#     print(f'roundness cutoff: {ROUNDNESS_CUTOFF}')
#     print(f'area cutoff: {AREA_CUTOFF}')
#     binary = ((store_agg['preds_detection_store'][idx] > 0)*255).astype(np.uint8)
#     img = img_store[idx]
#     gray_img = cv2.cvtColor(img.transpose((1,2,0)), cv2.COLOR_RGB2GRAY)
#
#     contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     if len(contours) > 0:
#         contour_fts = [get_contour_features(x, gray_img) for x in contours]
#
#         temp_df = pd.DataFrame(contour_fts)
#         temp_df.columns = ['cx','cy','eccentricity','area','roundness','solidity','intensity']
#         temp_df['file_id'] = ids_store[idx]
#         temp_df['prob_tumor'] = labels_store[idx]
#
#     im = np.zeros((256,256,3))
#
#     print('Magenta: small, round, darker \n Teal: small, round, brighter \n Orange: rest')
#     crit0 = (temp_df.area < AREA_CUTOFF) & (temp_df.intensity < INTENSITY_CUTOFF) & (temp_df.roundness < ROUNDNESS_CUTOFF)
#     for contour_idx in np.where(crit0)[0]:
#         cv2.fillPoly(im, pts = [contours[contour_idx]], color=(150,0,150))
#
#     crit1 = (temp_df.area < AREA_CUTOFF) & (temp_df.intensity > INTENSITY_CUTOFF) & (temp_df.roundness < ROUNDNESS_CUTOFF)
#     for contour_idx in np.where(crit1)[0]:
#         cv2.fillPoly(im, pts = [contours[contour_idx]], color=(0,150,150))
#
#     crit2 = ~(crit0 | crit1)
#     for contour_idx in np.where(crit2)[0]:
#         cv2.fillPoly(im, pts = [contours[contour_idx]], color=(150,50,0))
#
#     fig, axes = plt.subplots(1,2)
#     axes[0].imshow(im.astype(np.uint8))
#     axes[1].imshow(img_store[idx].transpose((1,2,0)))
#     plt.show()
