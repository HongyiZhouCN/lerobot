#!/bin/bash

#SBATCH -p accelerated # accelerated-h100 # dev_accelerated
#SBATCH -A hk-project-sustainebot
#SBATCH -J DP-training-Insertion

# Cluster Settings
#SBATCH -n 1       # Number of tasks
#SBATCH -c 16  # Number of cores per task
#SBATCH -t 2-00:00:00 # 30:00 # 2-00:00:00 # 1:00:00 # 2-00:00:00 ## # 06:00:00 # 1-00:30:00 # 2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1

export MUJOCO_GL=egl

python /hkfs/work/workspace/scratch/lx7270-flower/Codes/lerobot/lerobot/scripts/train.py
