#!/bin/bash
# FILENAME: job2.sh
module load anaconda/2020.11-py38
pip install beartype
cd $SLURM_SUBMIT_DIR
cd ./Blume-Capel
export WORLD_SIZE=NUM_GPUS
export NODE_RANK=RANK
export MASTER_ADDR=MASTER_NODE_IP
export MASTER_PORT=PORT
python -m torch.distributed.launch --nproc_per_node=2 BVAE_0.py
