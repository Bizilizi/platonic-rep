# worker_hvd_queue.py
import argparse, json, os, time, traceback
from pathlib import Path

import horovod.torch as hvd  # or horovod.tensorflow as hvd if TF
from mpi4py import MPI  # to coordinate the dynamic task queue

import torch


# ---- A placeholder "experiment runner" ----
def run_experiment_on_gpu(exp: dict, workdir: Path):
    """
    Replace this with your real code. 'exp' is a dict describing the experiment.
    Use torch / TF inside as needed; the process already has its own GPU.
    """
    # Example: pretend to do some GPU work
    x = torch.randn(4096, 4096)
    y = torch.matmul(x, x)
    # Write a dummy result file:
    out = workdir / f"exp_{exp['id']}_done.txt"
    out.write_text(
        f"finished {exp['id']} with param={exp.get('param')} ; sum={y.sum().item():.4f}\n"
    )


# ---- Dynamic task queue over MPI ----
TAG_READY = 1
TAG_DONE = 2
TAG_TASK = 3
TAG_STOP = 4


def master(comm, experiments, workdir: Path):
    size = comm.Get_size()
    num_workers = size - 1
    task_idx = 0
    closed_workers = 0

    print(f"[master] dispatching {len(experiments)} tasks to {num_workers} workers")

    status = MPI.Status()
    while closed_workers < num_workers:
        data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        source = status.Get_source()
        tag = status.Get_tag()

        if tag == TAG_READY:
            # worker is asking for a task
            if task_idx < len(experiments):
                comm.send(experiments[task_idx], dest=source, tag=TAG_TASK)
                task_idx += 1
            else:
                comm.send(None, dest=source, tag=TAG_STOP)
                closed_workers += 1

    print("[master] all workers closed")


def worker(comm, rank, workdir: Path):
    # signal readiness, then loop receiving tasks
    comm.send(None, dest=0, tag=TAG_READY)
    status = MPI.Status()
    while True:
        exp = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()
        if tag == TAG_TASK:
            try:
                run_experiment_on_gpu(exp, workdir)
            except Exception:
                traceback.print_exc()
            finally:
                # tell master we finished and ask for another
                comm.send(None, dest=0, tag=TAG_DONE)
                comm.send(None, dest=0, tag=TAG_READY)
        elif tag == TAG_STOP:
            break

    print(f"[worker {rank}] all tasks completed")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", required=True)
    parser.add_argument("--workdir", required=True)
    args = parser.parse_args()

    # --- Horovod init & GPU pinning (1 process == 1 GPU)
    hvd.init()
    if torch.cuda.is_available():
        torch.cuda.set_device(hvd.local_rank())  # Horovod-recommended pinning
        # Optional: limit threads per process if you use CPU-heavy dataloaders
        os.environ.setdefault("OMP_NUM_THREADS", "4")

    comm = MPI.COMM_WORLD
    rank = hvd.rank()
    size = hvd.size()

    # Rank 0 will load the task list and coordinate
    if rank == 0:
        experiments = [{"id": 0, "param": "test"}] * 20
        # In case experiments < workers, still safe: workers will receive STOP.
        master(comm, experiments, Path(args.workdir))
        # Close workers explicitly
        for dest in range(1, size):
            comm.send(None, dest=dest, tag=TAG_STOP)
    else:
        worker(comm, rank, Path(args.workdir))

    # Optional: barrier so all ranks finish together
    comm.Barrier()


if __name__ == "__main__":
    main()
