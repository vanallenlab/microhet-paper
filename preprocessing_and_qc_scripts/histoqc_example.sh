#! /bin/bash

# running with the operations in the regular `config.ini` but omitting the fat-like tissue and blur detection modules
python qc_pipeline.py -o ~/histoqc_masks/20210212_cm009 --config config_full_greenblueblack_sep_pen.ini /home/nyman/bms_images/svs-CA209-009/*/*svs --n 16
