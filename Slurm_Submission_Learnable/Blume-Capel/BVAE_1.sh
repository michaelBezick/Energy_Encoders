#!/bin/bash
# FILENAME: job2.sh
cd $SLURM_SUBMIT_DIR
cd ./Blume-Capel
module load anaconda/2020.11-py38
pip install beartype
export WORLD_SIZE=NUM_GPUS
export NODE_RANK=RANK
export MASTER_ADDR=MASTER_NODE_IP
export MASTER_PORT=PORT
python -m torch.distributed.launch --nproc_per_node=2 BVAE_1.py
