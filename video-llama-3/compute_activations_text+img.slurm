#!/bin/sh
#SBATCH --job-name="videollama-3-activations"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:1,VRAM:48G
#SBATCH --partition=DEADLINE
#SBATCH --comment="Results presentatiton"
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/zverev/logs/video-llama-3/activations/slurm-%j.out
#SBATCH --error=/storage/slurm/zverev/logs/video-llama-3/activations/slurm-%j.out
#SBATCH --nodelist=node19,node20,node21

nvidia-smi

source activate videollama3
set -x

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

# Run the script on each node, assigning each task to a different GPU
SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=0 \
    --jobid $SLURM_JOB_ID \
    "

srun $SRUN_ARGS bash -c "python compute_activations_text+img.py"