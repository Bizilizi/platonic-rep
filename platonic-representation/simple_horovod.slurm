#!/bin/bash
#SBATCH --job-name=exp-workers
#SBATCH --nodes=2                  # adjust
#SBATCH --ntasks-per-node=4        # 4 processes per node
#SBATCH --gpus-per-task=1          # 1 GPU per process
#SBATCH --cpus-per-task=8          # tune for your workload/dataloaders
#SBATCH --mem=0                    # or per-node memory, e.g., 0 = all
#SBATCH --time=04:00:00
#SBATCH --qos=mcml
#SBATCH --partition=mcml-dgx-a100-40x8,mcml-hgx-a100-80x4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-user=zverev@in.tum.de

# --- Environment (typical) ---
module switch intel-mpi openmpi/4.0.4-intel19

# Horovod uses NCCL for GPU collectives with PyTorch/TensorFlow
export HOROVOD_GPU_OPERATIONS=NCCL
export NCCL_DEBUG=INFO
# If needed on your fabric:
# export NCCL_SOCKET_IFNAME=^lo,docker0  # set to your real net interface
# export NCCL_IB_DISABLE=0               # if using IB

# SLURM + Open MPI: call `mpirun` once *inside* the allocation (don’t nest srun+mpirun).
# Open MPI auto-detects SLURM node allocation & task layout.  [oai_citation:2‡docs.open-mpi.org](https://docs.open-mpi.org/en/main/launching-apps/slurm.html?utm_source=chatgpt.com)
mpirun \
  --bind-to none --map-by slot \
  -x HOROVOD_GPU_OPERATIONS -x NCCL_DEBUG -x NCCL_SOCKET_IFNAME -x NCCL_IB_DISABLE \
  -x LD_LIBRARY_PATH -x PATH \
  python worker_hvd_queue.py \
    --experiments experiments.json \
    --workdir /path/to/outputs