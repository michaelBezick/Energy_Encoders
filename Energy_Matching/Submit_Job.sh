#!/bin/bash
sbatch --nodes=4 --gpus-per-node=2 --cpus-per-gpu=2 --constraint="B|D" --time=4:00:00 BVAE_0.sh
sbatch --nodes=4 --gpus-per-node=2 --cpus-per-gpu=2 --constraint="B|D" --time=4:00:00 BVAE_third_order.sh
