#!/bin/bash
# FILENAME: job2.sh
module load anaconda/2024.02-py311
pip install beartype
conda install cudnn=8.6
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install tensorflow
cd $SLURM_SUBMIT_DIR
python3 -u retraining.py
