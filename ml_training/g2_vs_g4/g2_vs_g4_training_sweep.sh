#! /bin/bash
TPS=250
CROP_SIZE=224
PATHS=/home/jupyter/20210310_inferred_tvnt_g2_vs_g4_profile_ccrcc_rcc_fixed_histoqc_masks__g2_g4_only____ptumor_geq070_tilecount_geq500.pkl

LABEL_VAR='g4_not_g2'
# SANITY_STEPS=-1 # check all val set 
SANITY_STEPS=0
EPOCHS=5
LR=5e-6

FOLDS=4
NUM_CLASSES=2
N_GPU=2
VAL_BATCH_FRAC=1.0
WORKERS=12

python /home/jupyter/mc_lightning/mc_lightning/data/prepare_data_folds_flexible_classes.py -df $PATHS --balance_var $LABEL_VAR --num_classes $NUM_CLASSES \
    --folds $FOLDS --out_dir .

BS=250
for FOLD in 0 1 2 3
do
    for AUG_STRENGTH in 0.5 1.0 1.5 4.0
    do
        OUT_SUB=./fold$FOLD\_aug_strength_$AUG_STRENGTH
        mkdir $OUT_SUB
        cp ./*fold$FOLD*.csv $OUT_SUB
        cp ./*test_*.csv $OUT_SUB
        python /home/jupyter/mc_lightning/mc_lightning/models/resnet/resnet_trainer.py --paths $PATHS --batch_size $BS --out_dir $OUT_SUB \
                --prefix '20x_512px_' --label_var $LABEL_VAR --crop_size $CROP_SIZE --gpus $N_GPU \
                --num_workers $WORKERS --precision 16 -tps $TPS --distributed_backend 'ddp' \
                --num_sanity_val_steps $SANITY_STEPS --max_epochs $EPOCHS \
                --lr $LR --row_log_interval 5 --fold_index $FOLD --default_root_dir $OUT_SUB \
                --limit_val_batches $VAL_BATCH_FRAC --val_check_interval 20 --aug_strength $AUG_STRENGTH
    done
done
