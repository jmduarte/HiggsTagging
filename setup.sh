#!/bin/bash
export HIGGSTAGGING=$PWD
export PYTHONPATH="$HIGGSTAGGING/modules:$PYTHONPATH"
export PYTHONPATH="$HIGGSTAGGING/train:$PYTHONPATH"
export PYTHONPATH="$HIGGSTAGGING/generator:$PYTHONPATH"
source activate HiggsTagging
