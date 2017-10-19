#!/bin/bash
source activate HiggsTagging
export HIGGSTAGGING=$PWD
export PYTHONPATH="$HIGGSTAGGING/modules:$PYTHONPATH"
export PYTHONPATH="$HIGGSTAGGING/train:$PYTHONPATH"
export PYTHONPATH="$HIGGSTAGGING/generator:$PYTHONPATH"
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH

