## Imports & Helpers
import sys
sys.path.append('../')
sys.path.append('/home/jupyter/checkmate-histo')

from checkmate_imports import *
import dask
from dask.distributed import Client


# global variables 
HUE_ORDER = ['stroma','pred_g2','intermediate_grade','pred_g4']
MIN_SEGMENT_SIZE = 50
NODE_DIFF_CUTOFF = 0.25
MIN_TIL_COUNT = 10
TIL_ISO_CUTOFF = 10
TIL_HIGH_CUTOFF = 25
TIL_AREA_CUTOFF = 50
FRAC_CUTOFF = 0.2
TILES_PER_MM2 = 0.256**-2

# assume 7x7 minimum case for a square area focus
# going 2 tiles inner would result in a 5x5 inner cube and thus area cutoff of 25
MIN_CENTER_AREA = 25


# https://github.com/dask/distributed/issues/2520
# Have to use __main__ convention when using dask Client in a script 
if __name__ == '__main__':
    
    ### Load saved files 
    assigned_df = pd.read_csv('/home/jupyter/checkmate-histo/consolidated_workflow/multislide_analysis/smoothed_tvnt_g2g4_inference_denize_multislide.csv')  

    client = Client()

    ### Run pipeline
    # pipeline specific params
    min_tumor_seg_mean = 0.7  # tried 0.8 but too strict; 0.7 appears to be a more natural breakpoint 

    to_run = assigned_df['unique_id'].unique()
    assigned_df = assigned_df.set_index(['unique_id','x','y'])
    
    store = []
    for uid in to_run:
        df = assigned_df.loc[uid].copy()
        big_future = client.scatter(df)
        f = client.submit(run_two_stage_watershed_segmentation, uid=uid, input_df=big_future, min_tumor_seg_mean=min_tumor_seg_mean)
        store.append(f)

    completed = []    
    for future, output in dask.distributed.as_completed(store, with_results = True, raise_errors=True):
        completed.append(output)

    torch.save(completed, f'/home/jupyter/checkmate-histo/consolidated_workflow/multislide_analysis/denize_multi_twostage_watershed_out_rerun.pkl')

    client.close()