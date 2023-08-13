# microhet-paper

---
This repository contains supporting code for "Spatially aware deep learning reveals tumor heterogeneity patterns that encode distinct kidney cancer states".

## Contents

```
additional_scripts
    Contains environment creation scripts and aggregate runner scripts for Mesmer/mIF analysis.
analysis_and_figure_creation_notebooks
    mif_analysis
        Multiplexed Immunofluorescence Analysis Files
    multislide_analysis
        Multiregion H&E Slide Analysis Files
    primary_analysis
        TCGA & CM-025 Primary Figures Analysis
    til_analysis
        TIL-specific Analysis Files 
ml_inference
    Scripts for inference using trained neural networks.
ml_training
    Scripts and Hyperparameters used for neural network training.
packages
    checkmate_imports.py
        Utility code used during general analysis
    image-graphs
        Utility code for creating and manipulating graphs 
    mc_lightning
        Toolkit developed in PyTorch Lightning for model training and inference
    vision-helpers
        Additional convenience and helper code 
preprocessing_and_qc_scripts
    Scripts used for HistoQC and tile generation
```
