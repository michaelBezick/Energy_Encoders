#!/bin/bash

sbatch --nodes=2 --gpus-per-node=2 --cpus-per-gpu=5 --constraint="B|D" --time=4:00:00 BVAE.sh