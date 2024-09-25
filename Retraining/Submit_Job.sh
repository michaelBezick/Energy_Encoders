#!/bin/bash
sbatch --nodes=1 --gpus-per-node=1 --cpus-per-gpu=20 --constraint="G|I|J|K|N" --time=4:00:00 ./retraining.sh
#sbatch --nodes=1 --gpus-per-node=1 --cpus-per-gpu=20 --constraint="B|D" --time=4:00:00 ./retraining.sh
