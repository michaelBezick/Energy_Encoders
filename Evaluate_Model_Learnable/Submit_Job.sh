#!/bin/bash
sbatch --nodes=1 --gpus-per-node=1 --cpus-per-gpu=1 --time=4:00:00 ./convert_to_KID.sh
