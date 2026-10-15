#!/bin/bash
# Train an ACT policy with LeRobot on the TU/e HPC cluster (one GPU).
#
# RECONSTRUCTED: the original job script was lost. This follows the settings recorded in
# the February 2026 runs (MrPsarik/PaperBalls, MrPsarik/PaperBallsExt train_config.json)
# and the notes from debugging it. Check partition and limits against the cluster docs
# before use.
#
# Submit:  sbatch slurm/train_act.slurm
# Watch:   squeue -u $USER    tail -f slurm-<jobid>.out

#SBATCH --job-name=act_train
#SBATCH --partition=tue.gpu.q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

# Batch jobs start in a non-interactive shell: conda is not set up unless we do it here.
source ~/.bashrc
conda activate lerobot

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # less fragmentation with AMP
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

nvidia-smi  # record which GPU the job got

# --num_workers=0: DataLoader workers crashed on these nodes ("DataLoader worker exited
# unexpectedly"), most likely because /dev/shm is too small for decoded video frames.
# Loading in the main process avoids shared memory at the cost of parallel loading.
lerobot-train \
  --dataset.repo_id=MrPsarik/PaperBalls \
  --policy.type=act \
  --policy.device=cuda \
  --policy.use_amp=true \
  --batch_size=12 \
  --num_workers=0 \
  --steps=120000 \
  --save_freq=5000 \
  --output_dir=outputs/train/paper_balls \
  --wandb.enable=true \
  --policy.repo_id=MrPsarik/PaperBalls
