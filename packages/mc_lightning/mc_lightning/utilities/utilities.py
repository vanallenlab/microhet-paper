import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
import pandas as pd


from PIL import Image

def pil_loader(path, bw = 'None'):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)

        if bw == 'None':
            return img.convert('RGB')
        elif bw == 'BW':
            thresh = 200
            fn = lambda x : 255 if x > thresh else 0
            r = img.convert('L').point(fn, mode='1')
            return r
        elif bw == 'GS':
            r = img.convert('1')
            return r

def create_cv_splits_id_only(path_df, num_folds=5, test_split=True, seed=None):
    """
    """
    if seed is not None:
        print('Using seed for CV splitting')
        np.random.seed(seed)

    sample_ids = path_df.index.unique().values
    splitter = KFold(n_splits=num_folds, shuffle=True)
    splitter.get_n_splits(sample_ids)
    splits = [x for x in splitter.split(sample_ids)]

    split_id_agg = []

    if test_split:
        for (train_idx, eval_idx) in splits:
            # split smaller eval. fold to provide dev and testing sets
            dev_idx, test_idx = train_test_split(eval_idx, test_size=0.5)

            # store dataframes for current fold
            split_id_agg.append(
                (sample_ids[train_idx], sample_ids[dev_idx], sample_ids[test_idx]))
    else:
        for (train_idx, eval_idx) in splits:
            split_id_agg.append((sample_ids[train_idx], sample_ids[eval_idx]))

    return split_id_agg


# def make_folds(paths_df_file, out_dir, balance_var, folds=4, seed=0, prefix='20x_512px_'):
#     try:
#         paths_df = pd.read_pickle(paths_df_file)
#     except:
#         paths_df = pd.read_csv(paths_df_file)
#
#     paths_df.index.name = 'idx'
#     data_anno = paths_df.reset_index().drop_duplicates('idx')
#
#     results = {}
#
#     if balance_var != 'None':
#         min_group_slides = data_anno.groupby(balance_var).full_path.count().min()
#         print('Min. number of slides: ', min_group_slides)
#         print('Balancing based on {}'.format(balance_var))
#         all_ids_subset = data_anno.groupby(balance_var).apply(
#             lambda x: x.sample(min_group_slides)).reset_index(0, drop=True)
#         all_ids_subset = all_ids_subset['idx'].values
#         paths_subset_df = paths_df.loc[all_ids_subset]
#
#         print(all_ids_subset.shape)
#
#         # updated splitting scheme
#         label0_splits = create_cv_splits_id_only(
#             paths_subset_df.loc[paths_subset_df[balance_var] == 0], num_folds=folds, seed=seed)
#         label1_splits = create_cv_splits_id_only(
#             paths_subset_df.loc[paths_subset_df[balance_var] == 1], num_folds=folds, seed=seed)
#
#         for split_idx, fold in enumerate(zip(label0_splits, label1_splits)):
#             train_ids = np.concatenate([fold[0][0], fold[1][0]])
#             eval_ids = np.concatenate([fold[0][1], fold[1][1]])
#
#             temp_path = os.path.join(out_dir, prefix + f'fold{split_idx}_train_slide_ids.csv')
#             pd.Series(train_ids).to_csv(temp_path)
#
#             temp_path = os.path.join(out_dir, prefix + f'fold{split_idx}_dev_slide_ids.csv')
#             pd.Series(eval_ids).to_csv(temp_path)
#
#             results
#     else:
#         print('Using all input slides')
#         all_ids_subset = data_anno.idx.values
#         paths_subset_df = paths_df
#         # create train/eval folds
#         id_splits = create_cv_splits_id_only(paths_subset_df, num_folds=folds)
#
#         for split_idx, (train_ids, eval_ids) in enumerate(id_splits):
#             temp_path = os.path.join(out_dir, prefix + f'fold{split_idx}_train_slide_ids.csv')
#             pd.Series(train_ids).to_csv(temp_path)
#
#             temp_path = os.path.join(out_dir, prefix + f'fold{split_idx}_dev_slide_ids.csv')
#             pd.Series(eval_ids).to_csv(temp_path)






## from prepare_data_folds_multitask -- has updated and more flexible way of setting up
def make_multitask_folds(paths_df_file, out_dir, task0_labels, task1_labels,
                         test_size=0.15, folds=4, seed=0, prefix='20x_512px_'):
    """
    Expects "meta_label" column to input DF which enumerates over the 4 possible label combinations across the 2 tasks
    (Each task 0/1 binary labels possible)

    :param paths_df_file:
    :param out_dir:
    :param task0_labels:
    :param task1_labels:
    :param folds:
    :param seed:
    :param prefix:
    :return:
    """
    # manual hardcoding
    try:
        paths_df = pd.read_pickle(paths_df_file)
    except:
        paths_df = pd.read_csv(paths_df_file)

    paths_df.index.name = 'idx'
    data_anno = paths_df.reset_index().drop_duplicates('idx')
    min_group_slides = data_anno.groupby([task0_labels, task1_labels]).full_path.count().min()
    print('Min. number of slides: ', min_group_slides)
    print('Balancing based on {}, {}'.format(task0_labels, task1_labels))
    all_ids_subset = data_anno.groupby([task0_labels, task1_labels]).apply(
        lambda x: x.sample(min_group_slides)).reset_index([0,1], drop=True)
    all_ids_subset = all_ids_subset['idx'].values
    paths_subset_df = paths_df.loc[all_ids_subset]

    print(all_ids_subset.shape)
    sample_ids = paths_subset_df.index.unique().values
    train_dev_ids, test_ids = train_test_split(sample_ids, test_size=test_size, shuffle=True)
    train_dev_df = paths_subset_df.loc[train_dev_ids]
    # updated splitting scheme
    train_dev_splits = [
        create_cv_splits_id_only(train_dev_df.loc[train_dev_df['meta_label'] == label], num_folds=folds,
                                 seed=seed) for label in range(4)
    ]

    for split_idx, fold in enumerate(zip(*train_dev_splits)):
        train_ids = np.concatenate([fold[label][0] for label in range(4)])
        eval_ids = np.concatenate([fold[label][1] for label in range(4)])

        temp_path = os.path.join(out_dir, prefix + f'fold{split_idx}_train_slide_ids.csv')
        pd.Series(train_ids).to_csv(temp_path)

        temp_path = os.path.join(out_dir, prefix + f'fold{split_idx}_dev_slide_ids.csv')
        pd.Series(eval_ids).to_csv(temp_path)

    if test_split:
        temp_path = os.path.join(out_dir, prefix + f'test_slide_ids.csv')
        pd.Series(test_ids).to_csv(temp_path)


## from prepare_data_folds_multitask -- has updated and more flexible way of setting up
def make_folds(paths_df_file, out_dir, label_name,
               test_size=0.15, folds=4, seed=0, num_classes=2, prefix='20x_512px_'):

    if seed is not None:
        print('Using seed for make_folds call')
        np.random.seed(seed)

    # manual hardcoding
    try:
        paths_df = pd.read_pickle(paths_df_file)
    except:
        paths_df = pd.read_csv(paths_df_file)

    paths_df.index.name = 'idx'
    data_anno = paths_df.reset_index().drop_duplicates('idx')
    min_group_slides = data_anno.groupby([label_name]).full_path.count().min()
    print('Min. number of slides: ', min_group_slides)
    print(f'Balancing based on {label_name}')
    all_ids_subset = data_anno.groupby(label_name).apply(lambda x: x.sample(min_group_slides)).reset_index(0, drop=True)
    all_ids_subset = all_ids_subset['idx'].values
    paths_subset_df = paths_df.loc[all_ids_subset]

    print(f'IDs remaining after balancing: {len(all_ids_subset)}')
    sample_ids = paths_subset_df.index.unique().values
    train_dev_ids, test_ids = train_test_split(sample_ids, test_size=test_size, shuffle=True)
    train_dev_df = paths_subset_df.loc[train_dev_ids]
    train_dev_splits = [
        create_cv_splits_id_only(train_dev_df.loc[train_dev_df[label_name] == label], num_folds=folds, seed=seed) for
        label in range(num_classes)]

    id_agg = {}
    id_agg['test_ids'] = test_ids
    id_agg['train_ids'] = {}
    id_agg['val_ids'] = {}

    for split_idx, fold in enumerate(zip(*train_dev_splits)):
        train_ids = np.concatenate([fold[label][0] for label in range(num_classes)])
        val_ids = np.concatenate([fold[label][1] for label in range(num_classes)])

        temp_path = os.path.join(out_dir, prefix + f'fold{split_idx}_train_slide_ids.csv')
        pd.Series(train_ids).to_csv(temp_path)

        temp_path = os.path.join(out_dir, prefix + f'fold{split_idx}_val_slide_ids.csv')
        pd.Series(val_ids).to_csv(temp_path)

        temp_path = os.path.join(out_dir, prefix + f'test_slide_ids.csv')
        pd.Series(test_ids).to_csv(temp_path)

        id_agg['train_ids'][split_idx] = train_ids
        id_agg['val_ids'][split_idx] = val_ids

    return id_agg








def get_checkpoint_and_data(output_dir, paths_file, fold_idx, model_idx, epoch=9):
    paths_df = pd.read_pickle(paths_file)
    root_dir = '/'.join(output_dir.split('/')[:-2])

    checkpoint_path =  os.path.join(output_dir
                                   ,f'checkpoint/statedict_fold{fold_idx}_model{model_idx}_{str(epoch+1).zfill(3)}.pth')
    checkpoint = torch.load(checkpoint_path)

    dev_ids = os.path.join(root_dir, f'20x_512px_fold{fold_idx}_dev_slide_ids.csv')
    dev_ids = pd.read_csv(dev_ids, header=None).iloc[:, 1].values
    dev_paths_df = paths_df.loc[dev_ids]

    loaded_model = torch.load(os.path.join(root_dir, 'saved_initialized_models.pth'))[model_idx]
    loaded_model.load_state_dict(checkpoint)

    train_transform, eval_transform = torch.load(os.path.join(root_dir ,'saved_transforms.pth'))[model_idx]

    return loaded_model, eval_transform, dev_paths_df


def normalize_attention_weights(weights):
    min_weight = weights.view(-1).min()
    max_weight = weights.view(-1).max()
    normed_weight = (weights - min_weight) / (max_weight - min_weight)
    return normed_weight


def visualize_tile_attention(tensor_batch, attention_weights, tile_labels, model_preds, num_examples=6,
                             normalize_attention=True):
    """
    plt.subplots utility
    modified to account for normalized input
    """
    mean = torch.Tensor([0.5, 0.5, 0.5])
    std = torch.Tensor([0.5, 0.5, 0.5])

    batch_size = tensor_batch.shape[0]

    num_to_plot = min(num_examples, batch_size)

    fig, axes = plt.subplots(2, num_to_plot)
    #     axes = axes.reshape(-1)
    for idx in range(num_to_plot):
        if normalize_attention:
            temp_weights = normalize_attention_weights(attention_weights[idx])
        else:
            temp_weights = attention_weights[idx]

        inp = tensor_batch[idx].permute(1, 2, 0)
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        axes[0, idx].imshow(inp)
        axes[1, idx].imshow(temp_weights)

        axes[0, idx].set_title(tile_labels[idx].item())
        axes[1, idx].set_title('Pred: {}'.format(model_preds[idx]))

        axes[0, idx].set_xticks([])
        axes[0, idx].set_yticks([])
        axes[1, idx].set_xticks([])
        axes[1, idx].set_yticks([])
    # fig.suptitle('Bag Label: {}'.format(label))
    return fig


def run_aggregation_multiitask(log_paths):
    expt_dirs = ['/'.join(path.split('/')[:-1]) for path in log_paths]
    expt_dirs_map = {int(path.split('/')[-3].split('_')[0]): path for path in expt_dirs}
    expt_names = [path.split('/')[-3] for path in expt_dirs]
    expt_key = {int(x.split('_')[0]): '_'.join(x.split('_')[1:]) for x in expt_names}
    agg_expts = expt_dirs

    tile_metrics = {}
    slide_metrics = {}

    for idx, path in expt_dirs_map.items():
        # tile level metrics
        tile_stats = pd.read_csv(os.path.join(path, 'tile_stats_all_models.csv'))
        temp_metrics = aggregate_results_multitask(tile_stats)
        temp_metrics['expt_label'] = expt_key[idx]
        tile_metrics[idx] = temp_metrics

        # slide level metrics
        slide_stats = pd.read_csv(os.path.join(path, 'slide_stats_all_models.csv'))
        slide_stats['task0_pred'] = (slide_stats['task0_pred'] > 0.5).astype(int)
        slide_stats['task1_pred'] = (slide_stats['task1_pred'] > 0.5).astype(int)
        temp_metrics = aggregate_results_multitask(slide_stats)
        temp_metrics['expt_label'] = expt_key[idx]
        slide_metrics[idx] = temp_metrics

    tile_metrics = pd.concat(tile_metrics.values())
    tile_metrics['min_class_acc'] = tile_metrics[['class0_acc', 'class1_acc']].apply(lambda x: x.min(), 1)
    tile_metrics['agg_level'] = 'tile'

    slide_metrics = pd.concat(slide_metrics.values())
    slide_metrics['min_class_acc'] = slide_metrics[['class0_acc', 'class1_acc']].apply(lambda x: x.min(), 1)
    slide_metrics['agg_level'] = 'slide'

    metrics = pd.concat([tile_metrics, slide_metrics])

    return metrics


def run_aggregation(log_paths):
    expt_dirs = ['/'.join(path.split('/')[:-1]) for path in log_paths]

    tile_stat_paths = {int(path.split('/')[-3].split('_')[0]): os.path.join(path, 'tile_stats_all_models.csv') for path
                       in expt_dirs}

    slide_stat_paths = {int(path.split('/')[-3].split('_')[0]): os.path.join(path, 'slide_stats_all_models.csv') for
                        path in expt_dirs}

    expt_dirs_map = {int(path.split('/')[-3].split('_')[0]): path for path in expt_dirs}

    expt_names = [path.split('/')[-3] for path in expt_dirs]

    expt_key = {int(x.split('_')[0]): '_'.join(x.split('_')[1:]) for x in expt_names}

    agg_expts = expt_dirs

    tile_metrics = {}

    for idx, path in expt_dirs_map.items():
        temp_metrics = aggregate_tile_results(path)
        temp_metrics['expt_label'] = expt_key[idx]
        tile_metrics[idx] = temp_metrics

    tile_metrics = pd.concat(tile_metrics.values())

    tile_metrics['min_class_acc'] = tile_metrics[['class0_acc', 'class1_acc']].apply(lambda x: x.min(), 1)
    tile_metrics['agg_level'] = 'tile'

    ### Also grab slide-level stats

    agg_expts = expt_dirs

    slide_metrics = {}

    for idx, path in expt_dirs_map.items():
        temp_metrics = aggregate_slide_results(path)
        temp_metrics['expt_label'] = expt_key[idx]
        slide_metrics[idx] = temp_metrics

    slide_metrics = pd.concat(slide_metrics.values())

    slide_metrics['min_class_acc'] = slide_metrics[['class0_acc', 'class1_acc']].apply(lambda x: x.min(), 1)
    slide_metrics['agg_level'] = 'slide'

    metrics = pd.concat([tile_metrics, slide_metrics])

    return metrics





def calc_per_class_tile_stats(df, label):
    temp = df.loc[df.label == label]
    return temp.correct_pred.mean()


def calc_metrics(df):
    f1 = f1_score(y_true=df.label.values, y_pred=df.pred.values)
    auroc = roc_auc_score(y_true=df.label.values, y_score=df.prob.values)
    auprc = average_precision_score(y_true=df.label.values, y_score=df.prob.values)
    class0_acc = calc_per_class_tile_stats(df, 0)
    class1_acc = calc_per_class_tile_stats(df, 1)

    return [f1, auroc, auprc, class0_acc, class1_acc]


def calc_accuracy_only(df):
    class0_acc = calc_per_class_tile_stats(df, 0)
    class1_acc = calc_per_class_tile_stats(df, 1)

    return [class0_acc, class1_acc]


def aggregate_tile_results(path):
    #     auc_stats = pd.read_csv(os.path.join(path,'/auc_stats_all_models.csv'))
    #     training_log = pd.read_csv(os.path.join(path,'/training_log_all_models.csv'))
    tile_stats = pd.read_csv(os.path.join(path, 'tile_stats_all_models.csv', ))

    eval_metrics = tile_stats.groupby(['model_idx', 'fold_idx', 'epoch']).apply(lambda x: calc_metrics(x))
    eval_metrics = pd.DataFrame(np.stack(eval_metrics.values), index=eval_metrics.index,
                                columns=['f1', 'auroc', 'auprc', 'class0_acc', 'class1_acc'])

    return eval_metrics


def aggregate_source_site_tile_results(path):
    tile_stats = pd.read_csv(os.path.join(path, 'tile_stats_all_models.csv', ))
    tile_stats['source_site'] = [x.split('-')[1] for x in tile_stats.slide_id.values]

    eval_metrics = tile_stats.groupby(['source_site', 'model_idx', 'fold_idx', 'epoch']).apply(
        lambda x: calc_accuracy_only(x))
    eval_metrics = pd.DataFrame(np.stack(eval_metrics.values), index=eval_metrics.index,
                                columns=['class0_acc', 'class1_acc'])

    return eval_metrics


def aggregate_slide_results(path):
    slide_stats = pd.read_csv(os.path.join(path, 'slide_stats_all_models.csv'))
    # binarize prediction for F1 score calculation
    slide_stats['pred'] = (slide_stats['pred'] > 0.5).astype(int)

    eval_metrics = slide_stats.groupby(['model_idx', 'fold_idx', 'epoch']).apply(lambda x: calc_metrics(x))
    eval_metrics = pd.DataFrame(np.stack(eval_metrics.values), index=eval_metrics.index,
                                columns=['f1', 'auroc', 'auprc', 'class0_acc', 'class1_acc'])

    return eval_metrics


def calc_per_class_tile_stats_multitask(df, label, task_idx=0):
    temp = df.loc[df[f'task{task_idx}_label'] == label]
    return temp[f'task{task_idx}_correct_pred'].mean()


def calc_metrics_multitask(df, task_idx=0):
    f1 = f1_score(y_true=df[f'task{task_idx}_label'].values, y_pred=df[f'task{task_idx}_pred'].values)
    auroc = roc_auc_score(y_true=df[f'task{task_idx}_label'].values, y_score=df[f'task{task_idx}_prob'].values)
    auprc = average_precision_score(y_true=df[f'task{task_idx}_label'].values,
                                    y_score=df[f'task{task_idx}_prob'].values)
    class0_acc = calc_per_class_tile_stats_multitask(df, 0, task_idx)
    class1_acc = calc_per_class_tile_stats_multitask(df, 1, task_idx)

    return [f1, auroc, auprc, class0_acc, class1_acc]


# def aggregate_tile_results_multitask(path):
#     tile_stats = pd.read_csv(os.path.join(path,'tile_stats_all_models.csv'))
#     slide_stats = pd.read_csv(os.path.join(path,'slide_stats_all_models.csv'))

def aggregate_results_multitask(df):
    metrics = []
    for task_idx in range(2):  # hardcoded for 2 tasks
        temp_metrics = df.groupby(['model_idx', 'fold_idx', 'epoch']).apply(
            lambda x: calc_metrics_multitask(x, task_idx))
        temp_metrics = pd.DataFrame(np.stack(temp_metrics.values), index=temp_metrics.index,
                                    columns=['f1', 'auroc', 'auprc', 'class0_acc', 'class1_acc'])
        temp_metrics['min_class_acc'] = temp_metrics[['class0_acc', 'class1_acc']].apply(lambda x: x.min(), 1)
        temp_metrics['task'] = task_idx
        metrics.append(temp_metrics)

    metrics = pd.concat(metrics)

    return metrics


def tile_sampler(x, tiles_per_slide):
    samples = x.sample(min(len(x), tiles_per_slide))
    return samples


def subsample_tiles(data_df, ids, tiles_per_slide, label_var, slide_var='slide_id'):
    # get subset dataframe
    subset_df = data_df.loc[ids]
    # perform subsampling
    subset_df = subset_df.reset_index().groupby(slide_var).apply(lambda x: tile_sampler(x, tiles_per_slide))
    subset_df = subset_df.reset_index(drop=True).dropna(subset=[label_var])

    return subset_df