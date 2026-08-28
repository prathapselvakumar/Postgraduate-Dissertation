# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

This is a fork of Apple's **LOOP (Leave-One-Out PPO)** reference implementation — a reinforcement-learning
system that trains LLM agents directly inside the [AppWorld](https://appworld.dev/) benchmark environment
(a stateful, multi-app simulated world). Paper: *Reinforcement Learning for Long-Horizon Interactive LLM
Agents* (arXiv:2502.01600). See [README.md](README.md) for the original usage docs.

This fork is being adapted as part of a Postgraduate Dissertation project (University of Manchester),
run on the CSF3 HPC cluster. Original Apple code carries `Copyright (C) 2025 Apple Inc.` headers — new
files/additions in this fork (e.g. `phi_agents/rl/vcc/`) do not. When editing existing Apple-authored
files, preserve their license headers.

## Environment setup

Two **separate** virtual environments are required (they have conflicting dependencies):

```bash
poetry install                                       # main env: training/inference (vLLM, torch, ray, ...)
python -m virtualenv appworld-env
appworld-env/bin/pip install click==8.2.1 appworld
appworld-env/bin/appworld install
export APPWORLD_ROOT=<path>
appworld-env/bin/appworld download data --root $APPWORLD_ROOT
```

`APPWORLD_ROOT` must be set in the environment for almost everything (training, eval, tests) to run.
Locally in this repo, `data/appworld_root` and `data/appworld_splits` hold the AppWorld data/task-split files.

Secrets (`HF_TOKEN`, `WANDB_API_KEY`) go in a git-ignored `ml-loop.env` file at the repo root, sourced
before training/sbatch jobs — not committed.

## Common commands

Run training locally (single 8-GPU node, GPUs 0-3 train / 4-7 infer):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
HF_TOKEN=<...> WANDB_API_KEY=<...> APPWORLD_ROOT=<...> PATH=<repo>/appworld-env/bin:$PATH \
accelerate launch --config_file ./phi_agents/rl/conf/accelerate_config.yaml --num_processes=4 \
  ./phi_agents/rl/train.py +global@_global_=appworld rl/gpu_allocation=four_learn_four_infer \
  llm=qwen_2_5_32b_train experiment_name=<name> rl.params.total_iterations=200 ...
```

Minimal single-GPU debug run (see [README.md](README.md#debugging) for the full flag list):

```bash
accelerate launch --config_file ./phi_agents/rl/conf/accelerate_config.yaml --num_processes=1 \
  ./phi_agents/rl/train.py +global@_global_=appworld rl/gpu_allocation=single_gpu \
  llm=qwen_2_5_7b_train wandb.enable=False rl.eval.enable=False rl.params.total_iterations=2
```

Evaluate a checkpoint on a task split:

```bash
python -m scripts.run_appworld_inference experiment_name=<name> llm=qwen_2_5_32b_eval \
  llm.adapter_path=<checkpoint> scenario_sampler.dataset_name=test_normal
python -m scripts.appworld_eval_parse_and_log experiment_name=<name> scenario_sampler.dataset_name=test_normal
```

Lint / type-check / test (from `.pre-commit-config.yaml` and `pyproject.toml`):

```bash
ruff check --fix phi_agents scripts tests web_api   # line-length 100, google docstrings
ruff format phi_agents scripts tests web_api
mypy --strict --ignore-missing-imports <path>
pytest                                              # markers: `notebook` runs jupyter notebooks
python scripts/test_vcc_pipeline.py                 # standalone smoke test for the VCBM credit extension
python scripts/torch_dist_healthcheck.py            # NCCL/distributed sanity check, run before big jobs
```

There is no dedicated `tests/` suite in this fork currently — `scripts/test_vcc_pipeline.py` is a
standalone script (not pytest-discovered) exercising `compute_credit_weights`, `compute_turn_token_spans`,
and a live `AppWorldInterface` execute-with-bookmark round trip; it requires `APPWORLD_ROOT` to be set.

## Running on CSF3 (University of Manchester HPC)

Full walkthrough in [scripts/README_CSF3.md](scripts/README_CSF3.md); quick reference in
[Csf3_terminal_commands](Csf3_terminal_commands) (SSH target, active job ID, conda env name, srun/sbatch
examples). Summary of the workflow:

1. `python scripts/sync_to_csf3.py` — archives the repo (excluding `.venv`, `appworld-env`, `logs`,
   checkpoints, `.git`) and rsyncs/extracts it to `~/scratch/.../ml-loop` on the cluster.
2. `bash scripts/setup_csf3.sh` (one-time) — installs Miniconda + a `ml-loop` conda env, points all HF/
   AppWorld caches at `~/scratch` (quota protection — home dir has none), installs Poetry deps and the
   separate `appworld-env`, pre-downloads Qwen 2.5 32B/7B.
3. `sbatch scripts/submit_loop_csf3.sbatch [gpu_allocation] [experiment_name]` — Slurm job; edit the
   `#SBATCH --gres=gpu:a100_80g:N` line to match the requested GPU allocation. Runs fully offline
   (`WANDB_MODE=offline`, `HF_HUB_OFFLINE=1`) and disables `torch.compile` (`TORCHDYNAMO_DISABLE=1`)
   because `nvcc` isn't on `PATH` on CSF3 compute nodes.
4. Monitor via `squeue -u $USER` / logs at `~/scratch/logs/<experiment_name>*.log`; checkpoints at
   `~/scratch/checkpoints/<experiment_name>`. **`~/scratch` auto-deletes files untouched for 3 months** —
   copy checkpoints out after training finishes.

## Architecture

### Training loop (`phi_agents/rl/train.py`)

`RLOOTrainer` is the core class. Each `accelerate`-launched process (one per learner GPU) runs the same
loop: sample scenarios → collect rollouts via vLLM → compute Leave-One-Out (or Leave-N-Out) advantage
baselines → compute a PPO-clipped or REINFORCE policy-gradient loss per rollout → backward/optimizer step
under FSDP2 → periodically checkpoint (LoRA adapter only) and evaluate.

- **No value network.** LOOP's memory efficiency comes from estimating the baseline as the average return
  of the *other* rollouts of the same scenario (`Baseline.LOO`/`Baseline.LNO` in `_compute_adv_estimates`),
  instead of training a critic.
- **LoRA + FSDP2**: the base model is wrapped with a PEFT LoRA adapter (`_setup_model_fsdp`); only the
  adapter is checkpointed (`_save_lora_checkpoint`), keeping checkpoints small and restart cheap.
- **Rollout filtering**: `_filter_rollouts` drops near-zero-advantage rollouts (`abs_adv_threshold`) and
  reshapes the remainder into equal-sized per-worker minibatches so all ranks step in sync.
- **Importance-weighted loss**: because rollouts are generated by vLLM (a slightly different numerical
  path than the FSDP/PyTorch training forward pass), `_loss`/`_surrogate_loss` recompute log-probs under
  the current model and apply a trajectory- or token-level importance-weight-corrected PG/PPO objective.
  Divergence between the vLLM-time and recompute-time log-probs is tracked as a KL estimate; runs abort
  the iteration (`stop_iteration`) on excessive KL or NaN loss, and hard-fail after 20 consecutive skips.
- Config is Hydra-based (`phi_agents/rl/config.py::get_config`, composed from `phi_agents/rl/conf/`).
  `+global@_global_=appworld` (see `phi_agents/rl/conf/global/appworld.yaml`) pulls in the AppWorld-specific
  defaults (scenario sampler, scenario runner, agent config) on top of the base `config.yaml` defaults list
  (optimizer, gpu_allocation, params, llm).

### Distributed execution (Ray)

The system runs as a single-node Ray cluster (`phi_agents/rl/utils/ray_utils.py::connect_ray_cluster`)
with three logical worker roles co-located on the same node(s): the **trainer** (the `accelerate`
processes running `train.py`), the **vLLM server(s)** (tagged with the custom Ray resource
`VLLM_RESOURCE = "vllm_server"` so work can be scheduled specifically onto GPUs hosting an inference
server), and the **evaluator** (`phi_agents/rl/eval.py`, driven asynchronously via
`phi_agents/rl/callbacks.py::EvalCallback` so validation doesn't block training). `gpu_allocation` configs
under `phi_agents/rl/conf/rl/gpu_allocation/` (e.g. `four_learn_four_infer`, `two_shared`, `single_gpu`)
control how GPUs are split between learning and inference roles, including the "shared" configs where
training and vLLM inference time-share the same GPUs (see `compute_gpu_mem_utilization` in `train.py` for
how memory is partitioned in that case).

### Rollout collection (`VLLMRolloutWorker`, `ParallelScenarioSampler`, `appworld_scenario_runner.py`)

`ParallelScenarioSampler` draws AppWorld task scenarios (thread-pooled) and hands them to
`VLLMRolloutWorker`, which drives the vLLM inference server(s) to actually play out each scenario as a
`TrainingRollout` (see `phi_agents/rl/type_defs.py`: `Scenario`/`ScenarioRunner` ABCs,
`TrainingRollout` = messages + return + `PolicyTokenInfo` token/logprob/is_output arrays used for the loss).
The AppWorld-specific agent loop lives in `phi_agents/agent/minimal_vllm_react_agent.py` — a ReAct-style
loop (`phi_agents/agent/react_template.py` holds the prompt template and per-app API descriptions) that
alternates assistant Python-code messages with `ipython` execution-result messages against a live AppWorld
task, via `phi_agents/appworld/interface.py` (`AppWorldInterface`, wraps the `appworld` package installed
in the separate `appworld-env`) and `phi_agents/appworld/server.py`.

### VCBM — Visual Causal Chain Bookmarking (`phi_agents/rl/vcc/`, dissertation addition)

`phi_agents/rl/vcc/` is a fork-local addition (not part of the original Apple release)
implementing the dissertation's VCBM credit-assignment mechanism. It is the **default**
credit-assignment path for any rollout with at least one bookmarked turn — there is no
`use_vcc_credit` toggle; rollouts with zero bookmarks fall back to plain LOOP's uniform
trajectory-level advantage.

- **Causal event detection (φ_http)**: turns flagged as "bookmarks" are state-changing AppWorld
  API calls (as opposed to read-only/comment code), detected with zero extra HTTP round trips via
  `AppWorldInterface.execute_with_bookmark` → the AppWorld server's `/execute_with_bookmark` route
  (HTTP-method interception, not state-snapshot diffing).
- **State encoder (g_φ)**: `phi_agents/rl/vcc/state_encoder.py::encode_state` — a dependency-free,
  deterministic hashing-trick bag-of-words embedding over each turn's `ipython` observation text.
- **Episodic memory bank**: `phi_agents/rl/vcc/memory_bank.py::build_memory_bank` stores one
  `CausalEvent(turn_idx, key, action_return)` per bookmarked turn.
- **Retrieval + weighting**: `retrieve_and_weight` computes cosine similarity between the terminal
  observation and each memory key, retrieves the top-`rl.vcc_top_k` events, and assigns
  temperature-scaled (`rl.vcc_temperature`) softmax weights over the retrieved set. These map to
  per-token weights via `credit_weights.py::compute_retrieval_credit_weights` (non-retrieved turns
  get the `rl.vcc_alpha` floor weight, default `0.1`). When a rollout has fewer bookmarked turns
  than `vcc_top_k`, retrieval degenerates to "use all bookmarks" and the code instead calls the
  simpler flat-floor `compute_credit_weights` directly (equivalent result, cheaper).
- **VCBM/LOOP blend**: `RLOOTrainer._vcbm_weights`/`_loss` (`train.py`) combine the retrieval
  weights with the plain LOOP advantage as `(1 - rl.vcc_blend_alpha) * A_LOOP + rl.vcc_blend_alpha
  * A_VCBM`; `vcc_blend_alpha` defaults to `1.0` (pure VCBM) but can be set to `0.0` to recover
  plain LOOP for ablations.

Bookmark data flows from `episode.turn_bookmarks`/`turn_token_spans`, carried on
`AppWorldTrainingRollout` (`phi_agents/evals/appworld_rollout_data.py`) and computed via
`compute_turn_token_spans` in `phi_agents/rl/appworld_scenario_runner.py`.
`scripts/test_vcc_pipeline.py` is the smoke test for this subsystem — the encoder/memory-bank/
retrieval/blend unit tests run standalone (no `APPWORLD_ROOT` needed); the final
`test_execute_with_bookmark` test needs `APPWORLD_ROOT` set. Run it after touching
credit-weighting logic.

### Directory map

- `phi_agents/rl/` — training loop, Hydra configs (`conf/`), Ray/FSDP2/vLLM-client utilities, the VCBM
  extension, RL type defs.
- `phi_agents/agent/` — the ReAct agent loop and prompt templates used during rollouts.
- `phi_agents/appworld/` — thin wrapper around the external `appworld` package/server.
- `phi_agents/vllm/` — vLLM server lifecycle management (start/stop/health, shared with inference).
- `phi_agents/evals/` — AppWorld evaluation/scoring and rollout logging (wandb, HTML rollout viewers).
- `phi_agents/inference/` — inference-time (non-training) config dataclasses (`LLMConfig`, `AppWorldConfig`).
- `phi_agents/utils/` — cross-cutting helpers: cloud/file I/O abstraction (`file_utils.py`, supports
  `file://` and cloud schemes), CUDA/memory profiling, distributed barriers, custom cross-entropy kernels
  (`cce.py`, memory-efficient forward pass used during training).
- `phi_agents/visualization/` — episode/rollout visualization (bokeh/plotly/matplotlib).
- `scripts/` — CLI entry points (inference, eval parsing, CSF3 sync/setup, health checks) invoked via
  `python -m scripts.<name>` with Hydra overrides as CLI args.

### Config composition (Hydra)

Configs live under `phi_agents/rl/conf/`. `config.yaml` is the training entry point with a `defaults` list
(`rl/optimization`, `rl/gpu_allocation`, `rl/params`, `llm`); `appworld_eval.yaml` is the eval entry point.
`+global@_global_=appworld` merges `conf/global/appworld.yaml` into the root config to add the
AppWorld-specific scenario sampler/runner/agent groups. Everything is overridable from the CLI using
standard Hydra dotted-path syntax (as seen throughout the sbatch script and README examples), e.g.
`llm.vllm_server.max_model_len=16384`, `rl.eval.overrides.rl.num_scenario_runners=12`.
