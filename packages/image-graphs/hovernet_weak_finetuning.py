from mc_lightning.models.resnet.resnet_dataset import SlideDataset
from mc_lightning.models.resnet.resnet_transforms import RGBTrainTransform, RGBEvalTransform

import random
import torch
import torch.nn as nn
from pathml.ml.hovernet import _dice_loss_nc_head, _ce_loss_nc_head


def get_pseudomask(preds_detection, prob_tumor):
    masks = []
    for pred, prob in zip(preds_detection, prob_tumor):
        ## get pseudo mask based on pretrained NP outputs
        nucl_bins = list(np.unique(pred))[1:]  # cut off 0 category (background)

        pseudo_sampling_size = int(prob * len(nucl_bins))
        z = random.sample(nucl_bins, k=pseudo_sampling_size)  # artificially assign labels based on inferred p(TvNT)
        y = pred
        pseudo_t = ((y > 0) & np.isin(y, z)) * 1
        pseudo_nt = ((y > 0) & ~np.isin(y, z)) * 2
        pseudo_mask = torch.Tensor(pseudo_t + pseudo_nt).long()

        ## push mask to one-hot format to match (B, n_classes, H, W)
        onehot_pseudo_mask = torch.nn.functional.one_hot(pseudo_mask, num_classes=3)
        onehot_pseudo_mask = onehot_pseudo_mask.permute(2, 0, 1)
        masks.append(onehot_pseudo_mask)

    return torch.stack(masks)


def get_pseudomask_noise(preds_detection, prob_tumor, noise_amount=0.25):
    masks = []
    for pred, prob in zip(preds_detection, prob_tumor):
        ## get pseudo mask based on pretrained NP outputs
        nucl_bins = list(np.unique(pred))[1:]  # cut off 0 category (background)

        if prob > noise_amount:
            noisy_prob = prob - np.random.uniform(low=0, high=noise_amount)
        else:
            noisy_prob = prob

        pseudo_sampling_size = int(noisy_prob * len(nucl_bins))
        z = random.sample(nucl_bins, k=pseudo_sampling_size)  # artificially assign labels based on inferred p(TvNT)
        y = pred
        pseudo_t = ((y > 0) & np.isin(y, z)) * 1
        pseudo_nt = ((y > 0) & ~np.isin(y, z)) * 2
        pseudo_mask = torch.Tensor(pseudo_t + pseudo_nt).long()

        ## push mask to one-hot format to match (B, n_classes, H, W)
        onehot_pseudo_mask = torch.nn.functional.one_hot(pseudo_mask, num_classes=3)
        onehot_pseudo_mask = onehot_pseudo_mask.permute(2, 0, 1)
        masks.append(onehot_pseudo_mask)

    return torch.stack(masks)



# # load the best model
# # checkpoint = torch.load("/mnt/disks/image_data/kidney/hover/hovernet_pannuke.pt", map_location='cpu')
# checkpoint = torch.load("/mnt/disks/image_data/kidney/hover/hovernet_pannuke.pt")

# n_classes_pannuke = 6

# # load the model
# hovernet = HoVerNet(n_classes=n_classes_pannuke)
# hovernet = torch.nn.DataParallel(hovernet)
# hovernet.load_state_dict(checkpoint)

# hovernet = hovernet.module
# hovernet.to('cuda');

# HEAD_ONLY = True
# if HEAD_ONLY:
#     print('only tuning a FC head layer')

# # ALTERNATIVE: start with pretrained NC
# if not HEAD_ONLY:
#     nc_branch_pretrain_clone = copy.deepcopy(hovernet.nc_branch)
#     new_decoder = nc_branch_pretrain_clone
#     new_decoder.to('cuda')

# # create new head layer to push NC to 3 classes
# decoder_post_fc = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1)
# decoder_post_fc.to('cuda');




# low_ptumor_subset_ids = pd.Series(low_ptumor.index.unique()).sample(25)

# high_ptumor_subset_ids = pd.Series(high_ptumor.index.unique()).sample(25)

# subset = paths.loc[list(low_ptumor_subset_ids) + list(high_ptumor_subset_ids)]

# TPS = 250
# SUBSAMPLE_SIZE = 1000
# balanced_subset = subset.groupby('file_id').apply(lambda x: x.sample(min(len(x), TPS))).reset_index(0, drop=True)
# balanced_subset = balanced_subset.sample(SUBSAMPLE_SIZE)

# paths_nonsubset = paths.drop(list(subset.index.unique())).sample(100)

# from torch.utils.data import DataLoader

# BATCH_SIZE = 10
# WORKERS = 8

# ft_dataset = SlideDataset(
#     paths=balanced_subset.full_path.values,
#     slide_ids=balanced_subset.index.values,
#     labels=balanced_subset.prob_tumor.values,
#     transform_compose=RGBEvalTransform(512, 256, norm_mean=[0,0,0], norm_std=[1,1,1])
# )

# ft_dataloader = DataLoader(ft_dataset, batch_size=BATCH_SIZE, num_workers=WORKERS, shuffle=True)

# val_dataset = SlideDataset(
#     paths=paths_nonsubset.full_path.values,
#     slide_ids=paths_nonsubset.index.values,
#     labels=paths_nonsubset.prob_tumor.values,
#     transform_compose=RGBEvalTransform(512, 256, norm_mean=[0,0,0], norm_std=[1,1,1])
# )

# val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=WORKERS, shuffle=True)

# # set up optimizer
# if HEAD_ONLY:
#     model_params = list(decoder_post_fc.parameters())
#     LR = 1e-3
# else:
#     model_params = list(new_decoder.parameters()) + list(decoder_post_fc.parameters())
#     LR = 1e-5

# opt = torch.optim.Adam(model_params, lr=LR)

# minibatch_train_losses = []
# PLOT_FREQ = 15
# NUM_VAL_VIS = 4
# set_rc(6,5)
# for epoch in range(2):
#     print(f'==== epoch {epoch} ====')
#     for batch_idx, (imgs, labels, ids) in enumerate(ft_dataloader):
#         # zero out gradient
#         opt.zero_grad()

#         ## use pretrained model to get initial outputs to work with
#         with torch.no_grad():
#             encoding = hovernet.encoder(imgs.cuda())
#             frozen_outputs = hovernet(imgs.cuda()) # TODO modify forward pass to give encoding and outputs?
#             if HEAD_ONLY:
#                 nc_branch_out = hovernet.nc_branch(encoding)  # decode

#         # get baseline nuclear segmentations
#         preds_detection, preds_classification = post_process_batch_hovernet(frozen_outputs, n_classes=6)

#         # generate pseudomasks
# #         pseudomasks = get_pseudomask(preds_detection, labels)
#         pseudomasks = get_pseudomask_noise(preds_detection, labels)

#         if not HEAD_ONLY:
#             # decode NC and transform with new fully tuned NC branch
#             nc_branch_out = new_decoder(encoding)  # decode

#         ft_nc = decoder_post_fc(nc_branch_out) # transform

#         # calc relevant losses and combine
#         dice_loss_term = _dice_loss_nc_head(ft_nc, pseudomasks.cuda())
#         crossent_loss_term = _ce_loss_nc_head(ft_nc, pseudomasks.cuda())
#         loss = dice_loss_term + crossent_loss_term

#         # track loss
#         minibatch_train_losses.append(loss.item())

#         # compute gradients
#         loss.backward()

#         # step optimizer and scheduler
#         opt.step()

#         if batch_idx % PLOT_FREQ == 0:
#             print(batch_idx, loss)
#             for val_idx, (imgs, labels, ids) in enumerate(val_dataloader):
#                 if val_idx == 0:
#                     ## use pretrained model to get initial outputs to work with
#                     with torch.no_grad():
#                         encoding = hovernet.encoder(imgs.cuda())
#                         nc_branch_out = hovernet.nc_branch(encoding)  # decode
#                         ft_nc = decoder_post_fc(nc_branch_out.cuda())

#                     fig, axes = plt.subplots(2,NUM_VAL_VIS)

#                     for sub_idx in range(NUM_VAL_VIS):
#                         axes[0, sub_idx].imshow(ft_nc[sub_idx].softmax(0).permute(1,2,0).detach().cpu().numpy())
#                         axes[1, sub_idx].imshow(imgs[sub_idx].permute(1,2,0))
#                     plt.show()