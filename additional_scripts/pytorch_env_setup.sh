#! /bin/bash

####### Minimal environment setup for updated Pytorch Lightning (2022/01/21)

#conda create --name pl 
#conda activate pl

conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch

pip install pytorch-lightning
conda install seaborn pandas

# adding jupyter kernel
conda install ipykernel 
ipython kernel install --name=pl --user 

# downgrade to legacy pl
pip install 'pytorch-lightning==1.2.4' --force-reinstall

# force appropriate versioning 
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
