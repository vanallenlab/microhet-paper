import argparse

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from torch.utils import data
from tqdm import tqdm

# my additions
import sys
import os
import copy
import configparser
from decimal import Decimal

sys.path.append('/home/nyman/')
from tmb_bot import utilities
from tmb_bot.utilities import class_balance_sampler, pil_loader
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append('/home/nyman/histopath-analysis')
#from generic_vae import pca_simple_vis
import datetime
from sklearn.decomposition import PCA
import seaborn as sns
import numpy as np 
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

def get_timestamp():
    timestamp = '_'.join(str(datetime.datetime.utcnow()).replace(
        ':', '.').replace('.', '-').split(' '))
    return timestamp


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


def supervised_train_loop(epoch, loader, model, forward_loss, optimizer, criterion, scheduler, log):
    """
    Expects model to have a single output (likely logits) that are passed to something like nn.CrossEntropyLoss()
    Expects model forward to only take minibatch of input images

    :param epoch:
    :param loader:
    :param model:
    :param optimizer:
    :param criterion:
    :param scheduler:
    :param device:
    :param log:
    :return:
    """
    model.train()

    criterion = nn.CrossEntropyLoss()

    loss_sum = 0

    for batch_idx, (imgs, labels, slide_ids) in enumerate(loader):
        model.zero_grad()

        if torch.cuda.device_count() == 1:
            imgs = imgs.cuda()
            labels = labels.cuda()
        if forward_loss:  # loss calculated in forward pass
            out, loss = model(imgs, labels)
            loss = loss.mean()  # collapse multi GPU output format
        else:  # loss calculated after forward pass [will be all on GPU0]
            out = model(imgs)
            loss = criterion(out, labels.long().cuda())

        loss.backward()

        #if scheduler is not None:
        #    scheduler.step()

        optimizer.step()

        with torch.no_grad():
            loss_sum += loss

        lr = optimizer.param_groups[0]['lr']
        avg_loss = (loss_sum/(batch_idx+1)).item()
        # TODO give print frequency an argument
        if batch_idx % 50 == 0:
            print(f'Training: epoch {epoch}, batch {batch_idx}, lr {Decimal(lr):.2e}, running loss {avg_loss:.3f}')

        log.loc[(epoch, batch_idx), 'train_loss'] = loss.item()



def evaluate_slide_data_simplified(model, dataloader, forward_loss=False, model_idx=0, fold_idx=0, epoch=0):
    model.eval()

    label_store = []
    id_store = []
    pred_store = []

    with torch.no_grad():
        for batch_idx, (imgs, labels, slide_ids) in enumerate(dataloader):
            if forward_loss:
                (pred, attention_weights), loss = model(imgs, labels)
            else:
                pred = model(imgs)

            pred_store.append(pred.exp())  # assuming model produces logits
            label_store.append(labels)
            slide_ids = np.array(slide_ids)
            id_store.append(slide_ids)

    # create tile level stats aggregation DF
    results = pd.DataFrame(columns=['label', 'slide_id', 'pred'])
    label_agg = torch.cat(label_store)
    results['label'] = label_agg
    results['slide_id'] = np.concatenate(id_store)
    results['prob'] = torch.cat(pred_store).cpu().softmax(-1)[:,1]
    results['pred'] = torch.cat(pred_store).cpu().argmax(-1)
    results['correct_pred'] = (results.label == results.pred)

    # annotate experiment details
    results['model_idx'] = model_idx
    results['fold_idx'] = fold_idx
    results['epoch'] = epoch

    # get tile level AUC
    tile_auc = roc_auc_score(results.label, results.prob)
    print(f'\nEval: Model {model_idx}, Fold {fold_idx}, Epoch {epoch}')
    print('Tile accuracy (mean overall): ', results.correct_pred.mean())
    print('Tile level AUC: ', tile_auc)

    # get mean-aggregated AUC
    mean_results = results.groupby('slide_id').mean()
    slide_mean_auc = roc_auc_score(mean_results.label.astype(int), mean_results.prob)
    print('Slide-mean AUC: ', slide_mean_auc)
    print('Mean pred: ', results.pred.mean())
    print('Class Accuracy: \n', results.groupby(['label']).correct_pred.mean())

    # combine tile and slide level AUC stats
    auc_agg = pd.DataFrame(columns=['model_idx', 'fold_idx', 'epoch', 'tile_auc', 'slide_auc']).set_index(
        ['model_idx', 'fold_idx', 'epoch'])
    auc_agg.loc[(model_idx, fold_idx, epoch)] = [tile_auc, slide_mean_auc]

    all_results = {
        'tile_stats': results,
        'slide_stats': mean_results,
        'auc_stats': auc_agg,
    }

    model.train()

    return all_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--scheduler', default=None)
    parser.add_argument('-bs', '--batch_size', type=int, default=100)
    parser.add_argument('--augment', action="store_true")
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--full_size', type=int, help='Uncropped input tile size', default=512)
    parser.add_argument('--seed', type=int, default=None, metavar='N',
                        help='set a random seed for torch and numpy (default: None)')
    parser.add_argument('--paths_df', type=str, help='file path of dataframe with ids/tile paths')
    parser.add_argument('--train_ids', type=str, help='file path of list of IDs in train set')
    parser.add_argument('--dev_ids', type=str, help='file path of list of IDs in validation (dev) set')
    parser.add_argument('-tps', '--tiles_per_slide', type=int, default=50,
                        help='if specified, num. tiles to sample per slide')
    parser.add_argument('--balance_var', type=str, default='is_kirc',
                        help='paths_df column on which to groupby to evenly sample from')
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--multigpu', type=bool, default=True)
    parser.add_argument('--out_dir', '-out', type=str, default='./experiment_outputs/')
    parser.add_argument('--log_out', action='store_true', help='whether to write to stdout/stderr')
    parser.add_argument('--model_path', type=str, default='./saved_initialized_models.pth')
    parser.add_argument('--transform_path', type=str, default='./saved_transforms.pth')
    parser.add_argument('--train_type', type=str, default='supervised')
    parser.add_argument('--budget', type=int, default=0)
    # parser.add_argument('--forward_loss', type=list, help='whether loss is calculated in forward pass')  # removed in favor of INI file
    parser.add_argument('--eval_freq', type=int, default=1, help='num. epochs between dev set evaluation')
    parser.add_argument('--config', help='config INI file with info for each model')
    parser.add_argument('--folds', type=int)
    parser.add_argument('--prefix', type=str, default='20x_512px_', help='prefix to append to output ID csvs')
    parser.add_argument('--eval_train', action="store_true", help='whether to evaluate train set as well as validation')

    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config)
    NUM_MODELS = len(config.sections())


    #### taken care of by pl.seed_everything
    # # set random seed if specified
    # if args.seed is not None:
    #     torch.manual_seed(args.seed)
    #     np.random.seed(args.seed)
    #     print('using random seed {} during training'.format(args.seed))

    #### worth keeping?
    # create timestamped experiment output
    timestamp = get_timestamp()
    output_dir = args.out_dir

    expt_dir = os.path.join(output_dir, timestamp)
    print(expt_dir)
    try:
        os.mkdir(expt_dir)
        os.mkdir(os.path.join(expt_dir, 'checkpoint'))
        # os.mkdir(os.path.join(expt_dir, 'sample'))
    except:
        pass

        # set up logging
    # checkpoint_dir = os.path.join(expt_dir, 'checkpoint')
    #
    # if args.log_out:
    #     sys.stderr = open(os.path.join(expt_dir, 'stderr.txt'), 'w')
    #     sys.stdout = open(os.path.join(expt_dir, 'stdout.txt'), 'w')
    #
    # print(args)
    #
    # if args.cuda & torch.cuda.is_available():
    #     device = 'cuda'
    #
    # # read in paths data once
    # try:
    #     paths_df = pd.read_pickle(args.paths_df)
    # except:
    #     paths_df = pd.read_csv(args.paths_df)
    #     paths_df = paths_df.set_index(paths_df.columns[0])
    #
    # paths_df.index.name = 'idx'
    #
    # # import custom model classes to avoid namespace issues when loading pickled models
    # from spatial_attention_model_specification_resnet50 import spatialAttentionSingleEncoder
    #
    # # load pickled models and transforms
    # loaded_models = torch.load(args.model_path)
    # loaded_transforms = torch.load(args.transform_path)
    #
    # # set up logging and result aggregation
    # all_training_logs = []
    # all_result_stats = []

    for fold_idx in range(args.folds):
        print(f'\n\n====== Fold {fold_idx} =====')
        # load train/dev IDs for particular fold
        temp_path = os.path.join(args.out_dir, args.prefix + f'fold{fold_idx}_train_slide_ids.csv')
        train_ids = pd.read_csv(temp_path, header=None).iloc[:, 1].values

        temp_path = os.path.join(args.out_dir, args.prefix + f'fold{fold_idx}_dev_slide_ids.csv')
        dev_ids = pd.read_csv(temp_path, header=None).iloc[:, 1].values

        # create path DF subsets
        train_paths_df = paths_df.loc[train_ids]
        dev_paths_df = paths_df.loc[dev_ids]

        # TODO find a cleaner way to flag when we don't want to use this arg (instead of checking != -1) [Ignore]
        if args.tiles_per_slide != -1:
            num_true = args.tiles_per_slide
            num_false = args.tiles_per_slide
            pred_variable = args.balance_var

            train_paths_df = train_paths_df.reset_index().groupby('idx').apply(
                lambda x: class_balance_sampler(x, num_true, num_false, pred_variable))
            dev_paths_df = dev_paths_df.reset_index().groupby('idx').apply(
                lambda x: class_balance_sampler(x, num_true, num_false, pred_variable))
            
        train_paths_df = train_paths_df.reset_index(level=1, drop=True).dropna(subset=[args.balance_var])
        dev_paths_df = dev_paths_df.reset_index(level=1, drop=True).dropna(subset=[args.balance_var])

        for model_idx, model_store in loaded_models.items():
            model = copy.deepcopy(model_store)

            # TODO add model checkpoint /resume step here
            # TODO add args.resume_training and args.previous_epochs
            #if args.resume_training:
            #    try:
            #        os.mkdir(os.path.join(expt_dir, 'checkpoint_resume'))
            #    except:
            #        pass
#
#                checkpoint_dir = os.path.join(expt_dir, 'checkpoint_resume')
#                prev_state_dict_file = f'statedict_fold{fold_idx}_model{model_idx}_{str(args.previous_epochs).zfill(3)}.pth'
#                prev_state_dict = torch.load(prev_state_dict_file)
#                model.load_state_dict(prev_state_dict)

            if args.augment:
                # TODO consider using simCLR implementation random augmentation code
                print('Using augmentations (Resize, RC, RHF, CJ [defaults]), Normalize')
                train_transform = loaded_transforms[model_idx][0]
                dev_transform = loaded_transforms[model_idx][1]
            else:
                print('Not using augmentations [only Resize, CenterCrop, Normalize]')
                train_transform = loaded_transforms[model_idx][1]
                dev_transform = loaded_transforms[model_idx][1]

            # TODO consider allowing diff models to use diff batch sizes
            train_dataset = SlideDataset(
                paths=train_paths_df.full_path.values,
                slide_ids=train_paths_df.index.values,
                labels=train_paths_df[args.balance_var].values,
                transform_compose=train_transform
            )
            dev_dataset = SlideDataset(
                paths=dev_paths_df.full_path.values,
                slide_ids=dev_paths_df.index.values,
                labels=dev_paths_df[args.balance_var].values,
                transform_compose=dev_transform
            )
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False,
                                      num_workers=args.workers)
            dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False,
                                    num_workers=args.workers)

            print(f'Training Model {model_idx}, {type(model)}')
                training_log = pd.DataFrame(columns=['model_idx', 'epoch', 'batch', 'loss']).set_index(['epoch', 'batch'])

            # load current model onto GPU
            if args.cuda:
                if args.multigpu:
                    model = nn.DataParallel(model).to(device)
                else:
                    model = model.to(device)

            optimizer = optim.Adam(model.parameters(), lr=config.getfloat(str(model_idx), 'lr'))
            #optimizer = optim.Adam(model.parameters(), lr=args.lr)

            # TODO specify scheduler in args
            if args.scheduler is not None:
                raise NotImplementedError
                # scheduler = xxxxx
            else:
                scheduler = None

            remaining_updates = args.budget  # set up for budget update train loop
            forward_loss = config.getboolean(str(model_idx),'forward_loss')  # get flag for whether loss calc'd in forward pass
            #forward_loss = config[str(model_idx)]['forward_loss'] == 'True'  # get flag for whether loss calc'd in forward pass
            for i in range(args.epoch):
                # run training loop
                if args.train_type == 'supervised':
                    supervised_train_loop(
                        i, train_loader, model, forward_loss, optimizer, scheduler, device, training_log
                    )
                if args.train_type == 'budget_supervised':
                    remaining_updates = budget_supervised_train_loop(
                        i, train_loader, model, forward_loss, optimizer, scheduler, device, training_log, remaining_updates
                    )

                # evaluate model performance on validation data
                if i % args.eval_freq == 0:
                    print('\n evaluating validation data...')
                    eval_stats = evaluate_slide_data_simplified(model, dev_loader, forward_loss=forward_loss, model_idx=model_idx, fold_idx=fold_idx, epoch=i)
                    all_result_stats.append(eval_stats)

                    torch.save(eval_stats,
                               os.path.join(expt_dir, f'checkpoint/dev_eval_dict_fold{fold_idx}_model{model_idx}_{str(i+1).zfill(3)}.pth'))
                    
                    if args.eval_train:
                        print('\n evaluating train data...')
                        eval_stats = evaluate_slide_data_simplified(model, train_loader, forward_loss, model_idx, fold_idx, epoch)
                        torch.save(eval_stats,
                                   os.path.join(expt_dir, f'checkpoint/train_eval_dict_fold{fold_idx}_model{model_idx}_{str(i+1).zfill(3)}.pth'))

                # save model checkpoint
                if args.multigpu:
                    torch.save(
                        model.module.state_dict(),
                        os.path.join(expt_dir, f'checkpoint/statedict_fold{fold_idx}_model{model_idx}_{str(i + 1).zfill(3)}.pth')
                    )
                else:
                    torch.save(
                        model.state_dict(),
                        os.path.join(expt_dir, f'checkpoint/statedict_fold{fold_idx}_model{model_idx}_{str(i + 1).zfill(3)}.pth')
                    )
                # save training log
                training_log.to_csv(
                    os.path.join(expt_dir, f'checkpoint/training_log_fold{fold_idx}_model{model_idx}_{str(i + 1).zfill(3)}.csv'))

            # store training log
            training_log['model_idx'] = model_idx
            training_log['fold_idx'] = fold_idx
            all_training_logs.append(training_log)

            # clean up cuda memory before moving on to next model [not sure if overkill to detach before deleting!]
            torch.cuda.empty_cache()
            del model

    all_training_logs = pd.concat(all_training_logs)
    all_training_logs.to_csv(os.path.join(expt_dir, f'checkpoint/training_log_all_models.csv'))

    # save aggregated evaluation statistics DFs to file
    all_tile_stats = pd.concat([x['tile_stats'] for x in all_result_stats])
    all_tile_stats.to_csv(os.path.join(expt_dir, f'checkpoint/tile_stats_all_models.csv'))

    all_slide_stats = pd.concat([x['slide_stats'] for x in all_result_stats])
    all_slide_stats.to_csv(os.path.join(expt_dir, f'checkpoint/slide_stats_all_models.csv'))

    all_auc_stats = pd.concat([x['auc_stats'] for x in all_result_stats])
    all_auc_stats.to_csv(os.path.join(expt_dir, f'checkpoint/auc_stats_all_models.csv'))



