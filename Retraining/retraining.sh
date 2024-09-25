#!/bin/bash
# FILENAME: job2.sh
module load anaconda/2024.02-py311
cd $SLURM_SUBMIT_DIR
conda activate myenv
python3 -u comparison_retraining.py
