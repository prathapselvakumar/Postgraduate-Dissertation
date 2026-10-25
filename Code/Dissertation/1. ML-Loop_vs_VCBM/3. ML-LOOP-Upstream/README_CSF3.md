# ML-LOOP baseline (upstream Apple code)

This folder is the LOOP baseline for the 32B AppWorld comparison. It is Apple's
[ml-loop](https://github.com/apple/ml-loop) at commit `f14107a`, with infrastructure fixes
needed to run on the CSF3 cluster and **no VCBM code**. Nothing in it reads `vcc_*` settings,
calls the bookmark endpoint, or changes LOOP's advantage: every token gets the leave-one-out
advantage, as in the original release.

`1. LOOP-Baseline/` is not used for this: despite its name it contains the full VCBM code path,
and its published run was plain LOOP only because its AppWorld server returned HTTP 404 on
every bookmark request.

## Changes from upstream

Every change is in [CSF3_INFRA_PATCH.diff](CSF3_INFRA_PATCH.diff). None touches the RL
algorithm, the loss, the advantage, or the agent.

| File | Change | Why |
|---|---|---|
| `phi_agents/vllm/vllm_server.py` | Pass `--compilation-config={"level": 0}` to vLLM | CSF3 compute nodes have no `nvcc`, so torch.compile crashes |
| `phi_agents/rl/utils/download.py` | Load models from the local HF cache when offline | Jobs run with `HF_HUB_OFFLINE=1` |
| `phi_agents/rl/train.py` | Timeout on LoRA saves; poll eval completion across ranks instead of blocking; SIGTERM handler; configurable NCCL timeout | A slow scratch filesystem or stuck eval otherwise hangs all GPUs until the NCCL watchdog kills the job |
| `phi_agents/rl/callbacks.py` | Non-blocking eval polling and a ping timeout | Same hang |
| `phi_agents/rl/cloud_checkpointer.py` | Upload timeout; no redundant wait before a barrier | Same hang |
| `phi_agents/rl/vllm_rollout_worker.py` | Retry vLLM startup; fix the path-equality assertion | Transient startup failures on shared nodes; the upstream assertion compared against `paths[0] * len(paths)` |
| `phi_agents/rl/utils/fsdp2_utils.py` | Unwrap `functools.partial` auto-wrap policies | Compatibility with the installed accelerate |
| `phi_agents/utils/file_utils.py` | `copytree(..., dirs_exist_ok=True)` | Resuming from a checkpoint into an existing directory |
| `phi_agents/rl/conf/rl/gpu_allocation/{two,four}_shared.yaml` | New: training and vLLM share the same GPUs | CSF3 allows 4 GPUs per job; upstream's 32B layout needs 8 |
| `phi_agents/rl/conf/config.yaml`, `conf/global/appworld.yaml` | W&B entity and project names | Logging only |
| `scripts/torch_dist_healthcheck.py` | The fork's version | Used by the job script before training |

## Running

Submit from this folder (it needs `logs/` and an `appworld-env`; the job uses `appworld` on
PATH if `appworld-env` is absent, and the bookmark patch is irrelevant here):

```bash
sbatch scripts/submit_mlloop_32b_4gpu.sbatch 1          # seed 1
ITERS=2 sbatch scripts/submit_mlloop_32b_4gpu.sbatch 1  # smoke test
```

Every setting that is not part of VCBM matches `2. VCC-Experiment/scripts/submit_32b_4gpu.sbatch`,
so the two runs differ only in the credit-assignment code.
