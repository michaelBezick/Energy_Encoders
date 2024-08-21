#!/bin/bash
# FILENAME: job2.sh
module load anaconda/2020.11-py38
cd $SLURM_SUBMIT_DIR
python convert_to_KID_format.py
