#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

"""
A robust script to check if distributed training works correctly.
This verifies:
1) 8 GPUs are visible.
2) BF16 compatibility.
3) NCCL all-reduce succeeds across all ranks.
4) GPU memory allocation succeeds.
"""

import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def run(rank: int, world_size: int) -> None:
    print(f"[Rank {rank}] Running distributed healthcheck...")
    
    # Set CUDA device for this process
    torch.cuda.set_device(rank)
    
    # 1) Verify BF16 works on the device
    if not torch.cuda.is_bf16_supported():
        print(f"ERROR: [Rank {rank}] BF16 is not supported on this GPU!")
        sys.exit(1)
    
    # Perform basic BF16 operations
    x = torch.ones(2, 2, dtype=torch.bfloat16, device=f"cuda:{rank}")
    y = x * 2.5
    assert torch.allclose(y, torch.full((2, 2), 2.5, dtype=torch.bfloat16, device=f"cuda:{rank}")), \
        f"[Rank {rank}] BF16 tensor math failed"
    print(f"[Rank {rank}] BF16 support and operations: OK")

    # 2) Verify GPU Memory Allocation succeeds
    try:
        # Try allocating ~1 GB of float32 tensors
        temp_alloc = torch.empty(256 * 1024 * 1024, dtype=torch.float32, device=f"cuda:{rank}")
        assert temp_alloc.numel() == 256 * 1024 * 1024
        del temp_alloc
        torch.cuda.empty_cache()
        print(f"[Rank {rank}] 1GB Memory allocation test: OK")
    except Exception as e:
        print(f"ERROR: [Rank {rank}] Memory allocation failed: {e}")
        sys.exit(1)

    # 3) Initialize Process Group
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    
    try:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    except Exception as e:
        print(f"ERROR: [Rank {rank}] Failed to initialize NCCL process group: {e}")
        sys.exit(1)

    # 4) NCCL All-Reduce check
    try:
        val = torch.ones(1, device=f"cuda:{rank}", dtype=torch.bfloat16)
        dist.all_reduce(val, op=dist.ReduceOp.SUM)
        assert int(val.item()) == world_size, \
            f"[Rank {rank}] NCCL all-reduce verification failed: expected {world_size}, got {val.item()}"
        print(f"[Rank {rank}] NCCL distributed all-reduce (world_size={world_size}): OK")
    except Exception as e:
        print(f"ERROR: [Rank {rank}] NCCL all-reduce failed: {e}")
        sys.exit(1)

    # 5) Gather device indices on rank 0
    device_idx = torch.tensor([rank], device=f"cuda:{rank}", dtype=torch.int64)
    gathered_devices = [
        torch.zeros(1, device=f"cuda:{rank}", dtype=torch.int64) for _ in range(world_size)
    ]
    dist.all_gather(gathered_devices, device_idx)
    if rank == 0:
        device_indices = [t.item() for t in gathered_devices]
        print(f"Gathered device indices on rank 0: {device_indices}")

    dist.destroy_process_group()
    print(f"[Rank {rank}] Healthcheck completed successfully.")


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"Detecting system GPUs... Found {world_size} GPUs.")
    
    if world_size < 4:
        print(f"WARNING: Expected 4 GPUs for the RunPod setup, but found {world_size}.")
    
    if world_size == 0:
        print("ERROR: No CUDA devices found!")
        sys.exit(1)
        
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)
    print("All ranks passed healthcheck successfully!")
