sbatch --nodes=1 --gpus-per-node=1 --cpus-per-gpu=1 --constraint="B|D" --time=4:00:00 ./generate_dataset.sh
