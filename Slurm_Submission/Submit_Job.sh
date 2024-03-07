#!/bin/bash

sbatch --nodes=2 --gpus-per-node=2 --cpus-per-gpu=8 --constraint="B|D" --time=4:00:00 ./Blume-Capel/BVAE_0.sh
#sbatch --nodes=2 --gpus-per-node=2 --cpus-per-gpu=8 --constraint="B|D" --time=4:00:00 ./Blume-Capel/BVAE_1.sh
#sbatch --nodes=2 --gpus-per-node=2 --cpus-per-gpu=8 --constraint="B|D" --time=4:00:00 ./Blume-Capel/BVAE_2.sh
#sbatch --nodes=2 --gpus-per-node=2 --cpus-per-gpu=8 --constraint="B|D" --time=4:00:00 ./Blume-Capel/BVAE_3.sh
#sbatch --nodes=2 --gpus-per-node=2 --cpus-per-gpu=8 --constraint="B|D" --time=4:00:00 ./Blume-Capel/BVAE_5.sh
#sbatch --nodes=2 --gpus-per-node=2 --cpus-per-gpu=8 --constraint="B|D" --time=4:00:00 ./Blume-Capel/BVAE_10.sh

#sbatch --nodes=2 --gpus-per-node=2 --cpus-per-gpu=8 --constraint="B|D" --time=4:00:00 ./Potts/BVAE_0.sh
#sbatch --nodes=2 --gpus-per-node=2 --cpus-per-gpu=8 --constraint="B|D" --time=4:00:00 ./Potts/BVAE_1.sh
#sbatch --nodes=2 --gpus-per-node=2 --cpus-per-gpu=8 --constraint="B|D" --time=4:00:00 ./Potts/BVAE_2.sh
#sbatch --nodes=2 --gpus-per-node=2 --cpus-per-gpu=8 --constraint="B|D" --time=4:00:00 ./Potts/BVAE_3.sh
#sbatch --nodes=2 --gpus-per-node=2 --cpus-per-gpu=8 --constraint="B|D" --time=4:00:00 ./Potts/BVAE_5.sh
#sbatch --nodes=2 --gpus-per-node=2 --cpus-per-gpu=8 --constraint="B|D" --time=4:00:00 ./Potts/BVAE_10.sh
#
#sbatch --nodes=2 --gpus-per-node=2 --cpus-per-gpu=8 --constraint="B|D" --time=4:00:00 ./QUBO/BVAE_0.sh
#sbatch --nodes=2 --gpus-per-node=2 --cpus-per-gpu=8 --constraint="B|D" --time=4:00:00 ./QUBO/BVAE_1.sh
#sbatch --nodes=2 --gpus-per-node=2 --cpus-per-gpu=8 --constraint="B|D" --time=4:00:00 ./QUBO/BVAE_2.sh
#sbatch --nodes=2 --gpus-per-node=2 --cpus-per-gpu=8 --constraint="B|D" --time=4:00:00 ./QUBO/BVAE_3.sh
#sbatch --nodes=2 --gpus-per-node=2 --cpus-per-gpu=8 --constraint="B|D" --time=4:00:00 ./QUBO/BVAE_5.sh
#sbatch --nodes=2 --gpus-per-node=2 --cpus-per-gpu=8 --constraint="B|D" --time=4:00:00 ./QUBO/BVAE_10.sh
