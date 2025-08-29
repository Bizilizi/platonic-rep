import gc
import os
import argparse
import traceback
from pathlib import Path

from tqdm import trange

import torch

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision.models.feature_extraction import create_feature_extractor

from datasets import load_dataset
from tasks import get_models
from models import load_llm, load_tokenizer
import utils


import horovod.torch as hvd  # or horovod.tensorflow as hvd if TF
from mpi4py import MPI  # to coordinate the dynamic task queue

import torch


def extract_llm_features(llm_model_name, dataset, args):
    """
    Extracts features from language models.
    Args:
        filenames: list of language model names by huggingface identifiers
        dataset: huggingface dataset
        args: argparse arguments
    """

    texts = [str(x["text"][args.caption_idx]) for x in dataset]

    save_path = utils.to_feature_filename(
        args.output_dir,
        args.dataset,
        args.subset,
        llm_model_name,
        pool=args.pool,
        prompt=args.prompt,
        caption_idx=args.caption_idx,
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"\ndataset: \t{args.dataset}")
    print(f"subset:    \t{args.subset}")
    print(f"processing:\t{llm_model_name}")
    print(f"save_path: \t{save_path}")

    if os.path.exists(save_path) and not args.force_remake:
        print("file exists. skipping")
        return

    language_model = load_llm(
        llm_model_name,
        qlora=args.qlora,
        force_download=args.force_download,
        cache_dir=args.cache_dir,
    )
    llm_param_count = sum([p.numel() for p in language_model.parameters()])
    tokenizer = load_tokenizer(llm_model_name)

    tokens = tokenizer(texts, padding="longest", return_tensors="pt")
    llm_feats, losses, bpb_losses = [], [], []

    # hack to get around HF mapping data incorrectly when using model-parallel
    device = next(language_model.parameters()).device

    for i in trange(0, len(dataset), args.batch_size):
        # get embedding cuda device
        token_inputs = {
            k: v[i : i + args.batch_size].to(device).long() for (k, v) in tokens.items()
        }

        with torch.no_grad():
            if "olmo" in llm_model_name.lower():
                llm_output = language_model(
                    input_ids=token_inputs["input_ids"],
                    attention_mask=token_inputs["attention_mask"],
                    output_hidden_states=True,
                )
            else:
                llm_output = language_model(
                    input_ids=token_inputs["input_ids"],
                    attention_mask=token_inputs["attention_mask"],
                )

            loss, avg_loss = utils.cross_entropy_loss(token_inputs, llm_output)
            losses.extend(avg_loss.cpu())

            bpb = utils.cross_entropy_to_bits_per_unit(
                loss.cpu(), texts[i : i + args.batch_size], unit="byte"
            )
            bpb_losses.extend(bpb)

            # make sure to do all the processing in cpu to avoid memory problems
            if args.pool == "avg":
                feats = torch.stack(llm_output["hidden_states"]).permute(1, 0, 2, 3)
                mask = token_inputs["attention_mask"].unsqueeze(-1).unsqueeze(1)
                feats = (feats * mask).sum(2) / mask.sum(2)
            elif args.pool == "last":
                feats = [v[:, -1, :] for v in llm_output["hidden_states"]]
                feats = torch.stack(feats).permute(1, 0, 2)
            else:
                raise NotImplementedError(f"unknown pooling {args.pool}")
            llm_feats.append(feats.cpu())

    print(f"average loss:\t{torch.stack(losses).mean().item()}")
    save_dict = {
        "feats": torch.cat(llm_feats).cpu(),
        "num_params": llm_param_count,
        "mask": tokens["attention_mask"].cpu(),
        "loss": torch.stack(losses).mean(),
        "bpb": torch.stack(bpb_losses).mean(),
    }

    torch.save(save_dict, save_path)

    del language_model, tokenizer, llm_feats, llm_output
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()


def extract_lvm_features(lvm_model_name, dataset, args):
    """
    Extracts features from vision models.
    Args:
        filenames: list of vision model names by timm identifiers
        image_file_paths: list of image file paths
        args: argparse arguments
    """
    assert args.pool == "cls", "pooling is not supported for lvm features"
    assert "vit" in lvm_model_name, "only vision transformers are supported"

    save_path = utils.to_feature_filename(
        args.output_dir,
        args.dataset,
        args.subset,
        lvm_model_name,
        pool=args.pool,
        prompt=None,
        caption_idx=None,
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"\ndataset: \t{args.dataset}")
    print(f"subset:    \t{args.subset}")
    print(f"processing:\t{lvm_model_name}")
    print(f"save_path: \t{save_path}")

    if os.path.exists(save_path) and not args.force_remake:
        print("file exists. skipping")
        return

    vision_model = (
        timm.create_model(lvm_model_name, pretrained=True, cache_dir=args.cache_dir)
        .cuda()
        .eval()
    )
    lvm_param_count = sum([p.numel() for p in vision_model.parameters()])

    transform = create_transform(
        **resolve_data_config(vision_model.pretrained_cfg, model=vision_model)
    )

    if "vit" in lvm_model_name:
        return_nodes = [f"blocks.{i}.add_1" for i in range(len(vision_model.blocks))]
    else:
        raise NotImplementedError(f"unknown model {lvm_model_name}")

    vision_model = create_feature_extractor(vision_model, return_nodes=return_nodes)
    lvm_feats = []

    for i in trange(0, len(dataset), args.batch_size):
        with torch.no_grad():
            ims = torch.stack(
                [transform(dataset[j]["image"]) for j in range(i, i + args.batch_size)]
            ).cuda()
            lvm_output = vision_model(ims)

            if args.pool == "cls":
                feats = [v[:, 0, :] for v in lvm_output.values()]
                feats = torch.stack(feats).permute(1, 0, 2)

            lvm_feats.append(feats.cpu())

    torch.save(
        {"feats": torch.cat(lvm_feats), "num_params": lvm_param_count}, save_path
    )

    del vision_model, transform, lvm_feats, lvm_output
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()


# ---- Dynamic task queue over MPI ----
TAG_READY = 1
TAG_DONE = 2
TAG_TASK = 3
TAG_STOP = 4


def master(comm, args):
    # MPI variables
    size = comm.Get_size()
    num_workers = size - 1
    task_idx = 0
    closed_workers = 0

    # get models
    llm_models, lvm_models = get_models(args.modelset, modality=args.modality)

    print(f"[master] dispatching {len(llm_models)} tasks to {num_workers} workers")

    status = MPI.Status()
    while closed_workers < num_workers:
        comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        source = status.Get_source()
        tag = status.Get_tag()

        if tag == TAG_READY:
            # worker is asking for a task
            if task_idx < len(llm_models):
                comm.send(
                    {
                        "llm_model": llm_models[task_idx],
                        "lvm_model": lvm_models[task_idx],
                    },
                    dest=source,
                    tag=TAG_TASK,
                )
                task_idx += 1
            else:
                comm.send(None, dest=source, tag=TAG_STOP)
        elif tag == TAG_DONE:
            # a worker reports completion; could collect results/logs here
            pass

    print("[master] all workers closed")


def worker(comm, rank, args):
    # signal readiness, then loop receiving tasks
    comm.send(None, dest=0, tag=TAG_READY)
    status = MPI.Status()

    # load dataset once outside
    dataset = load_dataset(
        "wikimedia/wit_base", split="train", cache_dir=args.cache_dir
    )

    while True:
        exp = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()
        if tag == TAG_TASK:
            try:
                extract_llm_features(exp["llm_model"], dataset, args)
                extract_lvm_features(exp["lvm_model"], dataset, args)
            except Exception:
                traceback.print_exc()
            finally:
                # tell master we finished and ask for another
                comm.send(None, dest=0, tag=TAG_DONE)
                comm.send(None, dest=0, tag=TAG_READY)
        elif tag == TAG_STOP:
            break

    print(f"[worker {rank}] all tasks completed")


def init_horovod():
    # --- Horovod init & GPU pinning (1 process == 1 GPU)
    hvd.init()
    if torch.cuda.is_available():
        torch.cuda.set_device(hvd.local_rank())  # Horovod-recommended pinning
        # Optional: limit threads per process if you use CPU-heavy dataloaders
        # os.environ.setdefault("OMP_NUM_THREADS", "4")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--force_download", action="store_true")
    parser.add_argument("--force_remake", action="store_true")
    parser.add_argument("--num_samples", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--pool", type=str, default="avg", choices=["avg", "cls"])
    parser.add_argument("--prompt", action="store_true")
    parser.add_argument("--dataset", type=str, default="prh")
    parser.add_argument("--subset", type=str, default="wit_1024")
    parser.add_argument("--caption_idx", type=int, default=0)
    parser.add_argument("--modelset", type=str, default="val", choices=["val", "test"])
    parser.add_argument(
        "--modality", type=str, default="all", choices=["vision", "language", "all"]
    )
    parser.add_argument("--output_dir", type=str, default="./results/features")
    parser.add_argument("--qlora", action="store_true")
    parser.add_argument(
        "--cache_dir", type=str, default="/dss/mcmlscratch/09/di97duk/datasets/"
    )
    parser.add_argument(
        "--page",
        type=int,
        default=0,
        help="Page number for batched feature extraction (0-indexed)",
    )
    parser.add_argument(
        "--page_size",
        type=int,
        default=5000,
        help="Number of samples per page. If None, use all samples.",
    )

    args = parser.parse_args()

    if args.qlora:
        print(f"QLoRA is set to True. The alignment score will be slightly off.")

    init_horovod()

    comm = MPI.COMM_WORLD
    rank = hvd.rank()
    size = hvd.size()

    if rank == 0:
        master(comm, args)
        for dest in range(1, size):
            comm.send(None, dest=dest, tag=TAG_STOP)
    else:
        worker(comm, rank, args)

    # Optional: barrier so all ranks finish together
    comm.Barrier()
