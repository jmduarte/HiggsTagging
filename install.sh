#!/bin/bash
conda create --copy --name HiggsTagging python=2.7.13
conda install --name HiggsTagging --file HiggsTagging.conda 
source activate HiggsTagging
pip install -r HiggsTagging.pip
cp activateROOT.sh  $CONDA_PREFIX/etc/conda/activate.d/activateROOT.sh 

# for gpu:
wget https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.0.1-cp27-none-lin\
ux_x86_64.whl
pip install --ignore-installed  --upgrade tensorflow_gpu-1.0.1-cp27-none-linux_x86_64.whl
pip install setGPU
