import pandas as pd
import argparse
import os
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import numpy as np


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
    parser.add_argument('--test_split', action='store_true')
    args = parser.parse_args()

    # set random seed if specified
    if args.seed is not None:
        np.random.seed(args.seed)
        print('using random seed {} during data splitting'.format(args.seed))


    # manual hardcoding
    try:
        paths_df = pd.read_pickle(args.paths_df)
    except:
        paths_df = pd.read_csv(args.paths_df)

    paths_df.index.name = 'idx'
    data_anno = paths_df.reset_index().drop_duplicates('idx')
    #min_group_slides = data_anno.groupby(
    #    args.balance_var).full_path.count().min()
    #print('Min. number of slides: ', min_group_slides)
    if args.balance_var != 'None':
        min_group_slides = data_anno.groupby(args.balance_var).full_path.count().min()
        print('Min. number of slides: ', min_group_slides)
        print('Balancing based on {}'.format(args.balance_var))
        all_ids_subset = data_anno.groupby(args.balance_var).apply(
            lambda x: x.sample(min_group_slides)).reset_index(0, drop=True)
        all_ids_subset = all_ids_subset['idx'].values
        paths_subset_df = paths_df.loc[all_ids_subset]

        print(all_ids_subset.shape)

        # updated splitting scheme
        label0_splits = create_cv_splits_id_only(
            paths_subset_df.loc[paths_subset_df[args.balance_var] == 0], num_folds=args.folds, test_split=args.test_split, seed=args.seed)
        label1_splits = create_cv_splits_id_only(
            paths_subset_df.loc[paths_subset_df[args.balance_var] == 1], num_folds=args.folds, test_split=args.test_split, seed=args.seed)

        for split_idx, fold in enumerate(zip(label0_splits, label1_splits)):
            train_ids = np.concatenate([fold[0][0], fold[1][0]])
            eval_ids = np.concatenate([fold[0][1], fold[1][1]])

            temp_path = os.path.join(args.out_dir, args.prefix + f'fold{split_idx}_train_slide_ids.csv')
            pd.Series(train_ids).to_csv(temp_path)

            temp_path = os.path.join(args.out_dir, args.prefix + f'fold{split_idx}_val_slide_ids.csv')
            pd.Series(eval_ids).to_csv(temp_path)
    else:
        print('Using all input slides')
        all_ids_subset = data_anno.idx.values
        paths_subset_df = paths_df
        # create train/eval folds
        id_splits = create_cv_splits_id_only(paths_subset_df, num_folds=args.folds, test_split=args.test_split)

        for split_idx, (train_ids, eval_ids) in enumerate(id_splits):
            temp_path = os.path.join(args.out_dir, args.prefix + f'fold{split_idx}_train_slide_ids.csv')
            pd.Series(train_ids).to_csv(temp_path)

            temp_path = os.path.join(args.out_dir, args.prefix + f'fold{split_idx}_val_slide_ids.csv')
            pd.Series(eval_ids).to_csv(temp_path)
