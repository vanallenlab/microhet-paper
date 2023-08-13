#! /bin/bash


##########################################################################################################################################################
## first we ned to specify some basic hyperparameters
# tile sampled per slide
TPS=1000

# crop size used for augmentations
CROP_SIZE=224

# batch size
BS=250

# csv or pickle containing file path locations for all tiles and their corresponding labels and slide IDs
PATHS=/home/jupyter/20201117_inferred_tvnt_g2_vs_g4_all_profile_rcc_fixed_histoqc_masks__g2_g4_only__ptumor_geq070.pkl

# the column of $PATHS that is used for prediction task
LABEL_VAR='g4_not_g2'

# number of classes present in $LABEL_VAR
NUM_CLASSES=2

# if set to -1, this will run through the whole validation set once before starting training loop
# SANITY_STEPS=-1 # check all val set 
SANITY_STEPS=0

# standard trainer hyperparams
EPOCHS=5
LR=1e-6

# number of splits to make in the data for cross-validation folds
FOLDS=4

# number of GPU devices to use 
N_GPU=2

# what fraction of validation set to evaluate
VAL_BATCH_FRAC=1.0

# specify how large the test set should be; remove the flag if you don't want to split a test set out
TEST_SIZE=0.15

# how many workers to use in dataloader objects -- generally 1 per CPU core present.
WORKERS=24

##########################################################################################################################################################

## Now we will set up fixed ID splits for each of our folds
## This will write two CSV files per fold -- one specifying train set slide IDs, another validation set slide IDs
## Note that if you specify $TEST_SIZE, this will also save a separate set of test slide IDs that are chosen PRIOR to train/validation fold creation.
python /home/jupyter/mc_lightning/mc_lightning/data/prepare_data_folds_flexible_classes.py -df $PATHS --balance_var $LABEL_VAR --num_classes $NUM_CLASSES \
    --folds $FOLDS --out_dir . --test_size $TEST_SIZE


## Finally we can iterate over the created folds and run the trainer+eval loop
## Note that if you want to run the test set, specify the store_true flag `--run_test_set`; This seem to run into issues with the ddp backend and I am looking into it
for FOLD in 0 1 2 3
do
        OUT_SUB=./fold$FOLD  # make a subdirectory for this fold
        mkdir $OUT_SUB
        cp ./*fold$FOLD*.csv $OUT_SUB  # move the train and validation fold slide ID file to this subdirectory
        cp ./*test_*.csv $OUT_SUB  # move the test set IDs to this subdirectory

        # finetune!
        python /home/jupyter/mc_lightning/mc_lightning/models/resnet/resnet_trainer.py --paths $PATHS --batch_size $BS --out_dir $OUT_SUB \
                --prefix '20x_512px_' --label_var $LABEL_VAR --crop_size $CROP_SIZE --gpus $N_GPU \
                --num_workers $WORKERS --precision 16 -tps $TPS --distributed_backend 'ddp' \
                --num_sanity_val_steps $SANITY_STEPS --max_epochs $EPOCHS \
                --lr $LR --row_log_interval 5 --fold_index $FOLD --default_root_dir $OUT_SUB \
                --limit_val_batches $VAL_BATCH_FRAC 
done
