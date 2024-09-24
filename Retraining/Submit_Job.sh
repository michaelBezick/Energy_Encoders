#!/bin/bash
sbatch --nodes=1 --gpus-per-node=1 --cpus-per-gpu=20 --constraint="G|I|J|K|N|L" --time=4:00:00 ./retraining.sh
