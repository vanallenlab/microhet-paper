import pandas as pd
import argparse
import os
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import numpy as np
import torch


def make_folds(paths_df_file, out_dir, label_name,
               test_size=0.15, folds=4, seed=0, num_classes=2, prefix='20x_512px_'):
    """
    Performs patient level k-fold splitting, after optionally setting aside a set of test set patients.
    Uses a categorical variable to do balanced sampling such that each category has the same number of patients
    present in each fold. Splits are saved to CSV to be later read during training time.

    Note that label categories should be pushed to ints (starting at 0)

    :param paths_df_file: csv or pickle path with paths and annotation information; indexed by patient/slide ID
    :param out_dir: location to save slide split CSV's
    :param label_name: column of `paths_df_file` used to do balanced sampling (and later classification)
    :param test_size: Fraction to use for test set allocation prior to k-fold creation
    :param folds: number of folds to generate
    :param seed: random seed for fixing
    :param num_classes: Number of classes present in `label_name`
    :param prefix: file prefix to use for saving slide splits to CSV file
    :return: DataFrame with ID splits
    """
    print(f'Balancing based on {label_name}')
    print(f'Making folds for {num_classes} classes')
    if seed is not None:
        print('Using seed for make_folds call')
        np.random.seed(seed)

    # manual hardcoding
    try:
        paths_df = pd.read_pickle(paths_df_file)
    except:
        paths_df = pd.read_csv(paths_df_file, index_col=0)

    paths_df.index.name = 'idx'
    data_anno = paths_df.reset_index().drop_duplicates('idx')
    min_group_slides = data_anno.groupby([label_name]).full_path.count().min()
    print('Min. number of slides: ', min_group_slides)
    all_ids_subset = data_anno.groupby(label_name).apply(lambda x: x.sample(min_group_slides)).reset_index(0, drop=True)
    all_ids_subset = all_ids_subset['idx'].values
    paths_subset_df = paths_df.loc[all_ids_subset]

    print(f'IDs remaining after balancing: {len(all_ids_subset)}')
    sample_ids = paths_subset_df.index.unique().values
    if test_size is not None:
        train_dev_ids, test_ids = train_test_split(sample_ids, test_size=test_size, shuffle=True)
    else:
        train_dev_ids = sample_ids

    train_dev_df = paths_subset_df.loc[train_dev_ids]
    print('Making train + validation splits...')
    train_dev_splits = [
        create_cv_splits_id_only(train_dev_df.loc[train_dev_df[label_name] == label], num_folds=folds, seed=seed) for
        label in range(num_classes)]

    id_agg = {}
    if test_size is not None:
        print('Setting aside test set before making Train/Validation folds')
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

        if test_size is not None:
            temp_path = os.path.join(out_dir, prefix + f'test_slide_ids.csv')
            pd.Series(test_ids).to_csv(temp_path)

        id_agg['train_ids'][split_idx] = train_ids
        id_agg['val_ids'][split_idx] = val_ids

    return id_agg


def create_cv_splits_id_only(path_df, num_folds=5, test_split=True, seed=None):
    """
    Uses sklearn's train_test_split to evenly create folds for a single category of patients.

    Note, can be used on its own if the label distribution is already balanced or less control
    is needed over how balanced each fold is.

    :param path_df: DataFrame
    :param num_folds:
    :param test_split:
    :param seed:
    :return: Nested dictionary of patient IDs corresponding to each fold
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', '-o', type=str)
    parser.add_argument('--train_slides', type=int, default=10)
    parser.add_argument('--dev_slides', type=int, default=5)
    parser.add_argument('--seed', type=int, default=None, metavar='N',
                        help='set a random seed for torch and numpy (default: None)')
    parser.add_argument('-df', '--paths_df', type=str,
                        help='file path of dataframe with ids/tile paths [pickle or csv]')
    parser.add_argument('--balance_var', type=str, default='is_kirc',
                        help='paths_df column on which to groupby to evenly sample from')
    parser.add_argument('--prefix', type=str, default='20x_512px_',
                        help='prefix to append to output ID csvs')
    parser.add_argument('--folds', type=int)
    # parser.add_argument('--test_split', action='store_true')
    parser.add_argument('--test_size', type=float, help='Fraction of slides to use for test set')
    parser.add_argument('--num_classes', type=int, help='Number of unique categories in balance_var')
    args = parser.parse_args()

    print(args)
    folds = make_folds(
        paths_df_file=args.paths_df,
        out_dir=args.out_dir,
        label_name=args.balance_var,
        test_size=args.test_size,
        folds=args.folds,
        seed=args.seed,
        num_classes=args.num_classes,
        prefix=args.prefix
    )

    torch.save(folds, os.path.join(args.out_dir, args.prefix+'all_fold_splits.pt'))