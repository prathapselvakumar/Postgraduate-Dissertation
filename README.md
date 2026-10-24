# Visual Causal-chain Bookmarking: Retrieval-Weighted Credit Assignment for Reinforcement Learning from Delayed Reward

This repository holds the code, results, and written work for my MSc Robotics dissertation at the University of Manchester (2026). The dissertation studies Visual Causal-chain Bookmarking (VCBM), a family of credit-assignment methods for reinforcement learning (RL) that concentrate the learning signal on the decisions that actually determine an episode's return, instead of spreading it evenly across every step.

## Contents

1. [Research problem](#research-problem)
2. [Method](#method)
3. [Experiments and results](#experiments-and-results)
4. [Limitations](#limitations)
5. [Repository structure](#repository-structure)
6. [Running the code](#running-the-code)
7. [Acknowledgements](#acknowledgements)

## Research problem

When an RL agent receives its reward only at the end of a long episode, a trajectory-level update gives every action the same learning signal, whether or not that action influenced the outcome. An LLM agent working through a task in the AppWorld benchmark, for example, may make dozens of tool calls, of which only a few change anything that matters to the task. Averaging the delayed reward across all of them slows learning and can reinforce actions that merely happened to occur in successful episodes.

Two established approaches address this. Leave-one-out baselines, as in LOOP (Chen et al., 2025), reduce the variance of the return estimate by comparing rollouts of the same task, but still assign one advantage to every token. Return redistribution, as in RUDDER (Arjona-Medina et al., 2019), trains an LSTM to predict the final return from each prefix of the trajectory and hands out reward according to changes in that prediction, which works only as well as the LSTM learns.

The dissertation tests a third option: identify which steps are causally responsible for the outcome, and weight credit towards them.

## Method

Both versions of VCBM follow the same three stages: detect decision points that plausibly carry causal weight, store them as bookmarks in a memory, and weight the learning signal according to that memory. The mechanism at each stage depends on the setting.

| Stage | Tabular MDP (watch repair) | Language-based MDP (AppWorld) |
| :--- | :--- | :--- |
| Detect | A statistical test classifies each state component as action-controlled, noise, static, or clock | A structural rule marks tool calls that change the environment's state |
| Store | Per-bookmark return statistics, updated with Welford's online estimator | An episodic memory of bookmarked turns |
| Weight | Action values estimated directly at each bookmarked decision | Cosine-similarity retrieval over the memory gives temperature-scaled weights, blended with LOOP's leave-one-out advantage |

In the tabular setting, the environment is small enough to discover its causal structure from interaction data. In AppWorld, the state is natural language and only partly observed, so causal discovery is not tractable and the structural rule takes its place.

## Experiments and results

### 1. RUDDER versus VCBM on a delayed-reward tabular task

The watch-repair task comes from the RUDDER demonstration code. At the first step the agent sees a watch of one of four brands and decides whether to repair it. Nothing it does afterwards matters, but the reward arrives only 50 steps later. Repairing is profitable for two of the four brands, which gives a known optimal policy for evaluation (it is never used in training). Each method was trained with 20 random seeds until 95% of first decisions over a 750-episode moving window were correct.

| Method | Seeds converged | Episodes to converge, mean ± SD | Median | Range |
| :--- | :---: | ---: | ---: | ---: |
| RUDDER | 20 / 20 | 3,327.1 ± 2,010.9 | 2,829 | 1,603 to 10,936 |
| VCBM | 20 / 20 | 1,532.3 ± 1,428.0 | 1,204.5 | 1,196 to 7,757 |

VCBM needed fewer episodes to converge, and a two-sided Mann-Whitney U test found the difference significant (U = 19, p < 0.001, rank-biserial r = 0.91). In all 20 seeds, VCBM's classifier labelled the five state components exactly as the environment generates them: the repaired flag as action-controlled, the two transport counters as noise, the brand as static, and the timestep as a clock.

### 2. LOOP versus VCBM-blended LOOP on AppWorld

Both runs fine-tuned Qwen2.5-7B-Instruct with LoRA (rank 8, alpha 16) for 200 iterations (9,600 rollouts) on the AppWorld training split, using FSDP2 across two GPUs on the University of Manchester CSF3 cluster, AdamW at a constant learning rate of 5 × 10⁻⁵, a sparse per-episode reward, and up to 50 interactions per episode. The baseline was intended to use plain LOOP (β = 0); the comparison blended in VCBM credit (β = 1, τ = 0.5, top-k = 6, α = 0.05). Both arms share the same training code, in which β defaults to 1, and the baseline's launch script did not set it, so whether the baseline really ran with β = 0 still has to be confirmed from its logged configuration (`1. LOOP-Baseline/scripts/verify_published_runs.py` does this on CSF3).

All figures below are from the final logged training step, taken from the Weights & Biases summary export for each run. They come from that step's 48 training rollouts (12 scenarios), not a held-out split, and the two runs' final batches had different difficulty mixes: VCBM's contained 4 hard-task rollouts against LOOP's 8. At LOOP's mix, VCBM's per-difficulty rates give an overall success of 0.417 (+17.6%), and the difficulty-averaged success rate (0.257 vs. 0.333, +29.7%) does not depend on the mix.

| Metric | LOOP | VCBM | Change |
| :--- | ---: | ---: | ---: |
| Mean return | 0.6682 | 0.7300 | +9.3% |
| Success rate, all tasks | 0.3542 | 0.4583 | +29.4% |
| Success rate, difficulty 1 (easy) | 0.5833 | 0.5000 | −14.3% |
| Success rate, difficulty 2 (medium) | 0.1875 | 0.5000 | +166.7% |
| Success rate, difficulty 3 (hard) | 0.0000 | 0.0000 | none |
| Maximum importance weight | 3.97 | 3.09 | −22.2% |
| KL estimate | 0.561 | 0.583 | +4.1% |
| Mean output tokens per rollout | 2,001.6 | 1,625.2 | −18.8% |
| Return per 1,000 output tokens | 0.334 | 0.449 | +34.6% |

VCBM-blended LOOP had a higher return and success rate, with almost all of the gain on medium-difficulty tasks. Neither method solved a hard task, and LOOP did better on easy tasks. VCBM also used fewer tokens per rollout while succeeding more often, so the shorter rollouts reflect finishing tasks in fewer steps, not giving up early. The final-step KL estimate was the one stability measure that slightly favoured LOOP.

## Limitations

- The AppWorld comparison is a single run of each method, so none of its differences have been tested for statistical significance. Only the watch-repair result has been tested across seeds.
- The two AppWorld runs differ in three VCBM hyperparameters (τ, top-k, α). The baseline's code reads these whenever β > 0, so they only stop mattering once the baseline's β = 0 is confirmed. Either way the comparison is one hyperparameter setting, not a sweep.
- The 48 rollouts per iteration come from 12 scenarios × 4 rollouts in the launch script. The logged temperature of 0 is the evaluation sampler's override (`rl.eval.overrides.llm.temperature`); training sampled at temperature 1.0.
- The AppWorld memory bank is rebuilt for every rollout, so it carries nothing across rollouts or iterations. Rollouts with at most top-k bookmarks give each bookmark weight 1, while those with more give retrieved bookmarks softmax weights averaging 1/k, so credit becomes much less concentrated once a rollout passes k bookmarks. The runs did not log how many rollouts had bookmarks; the code now logs this as `vcbm/*` metrics.
- The AppWorld bookmark detector is a hand-written rule about tool calls. It may not carry over to tasks where the causally important actions are something else.
- Run-peak stability figures discussed in the dissertation were read from exported charts, not numeric logs, and carry an estimated 10% to 20% reading error. They are left out of the table above.

## Repository structure

```text
Postgraduate-Dissertation/
├── Code/
│   ├── Dissertation/
│   │   ├── 1. ML-Loop_vs_VCBM/
│   │   │   ├── 1. LOOP-Baseline/      LOOP training and evaluation on AppWorld
│   │   │   └── 2. VCC-Experiment/     The same pipeline with VCBM credit (phi_agents/rl/vcc/)
│   │   ├── 2. Rudder_vs_VCBM/
│   │   │   ├── 1. RUDDER-Baseline/    RUDDER on the watch-repair task
│   │   │   ├── 2. VCBM-Experiment/    VCBM on the watch-repair task
│   │   │   ├── 3. Outputs/            Watch-repair and Mars-rover run outputs
│   │   │   └── 4. Fair-Comparison/    Controlled re-run: matched exploration, MC baselines, held-out seeds
│   │   └── 3. Final Outputs/          Exported result graphs (PDF) for both comparisons
│   └── Research work/                 Git submodules: vcbm, VCBM-rudder-demonstration-code
└── Documentation/
    └── Dissertation Document/
        ├── 0. Risk Assessment/
        ├── 1. Dissertation Topic Proposal/
        ├── 2. Project Outline/
        ├── 3. Research Plan/
        ├── 4. Initial Draft/
        └── 5. Final Draft/            LaTeX source and PDF of the dissertation
```

The folder `2. VCC-Experiment` keeps the method's earlier working name, Visual Causal Chains (VCC).

## Running the code

### Watch-repair task

Each version is a standalone Python script.

```bash
cd "Code/Dissertation/2. Rudder_vs_VCBM/1. RUDDER-Baseline/1. Watch-Repair"
python watch_repair.py
```

Run `watch_repair.py` in `2. VCBM-Experiment/1. Watch-Repair` the same way for the VCBM version. The `README.md` in each folder gives the options.

### AppWorld

The AppWorld experiments use a fork of [apple/ml-loop](https://github.com/apple/ml-loop) and need multiple GPUs. Installation, training, and evaluation are described in the `README.md` inside each experiment folder, and running on the CSF3 cluster in `scripts/README_CSF3.md`.

To fetch the submodules under `Code/Research work/`:

```bash
git submodule update --init --recursive
```

## Acknowledgements

The AppWorld experiments build on Apple's [ml-loop](https://github.com/apple/ml-loop) implementation of LOOP and on the [AppWorld](https://appworld.dev/) benchmark. The watch-repair environment and RUDDER baseline come from the [RUDDER](https://github.com/ml-jku/rudder) project. Each forked component keeps its original licence.

## References

- Arjona-Medina, J. A., Gillhofer, M., Widrich, M., Unterthiner, T., Brandstetter, J., & Hochreiter, S. (2019). RUDDER: Return decomposition for delayed rewards. *Advances in Neural Information Processing Systems 32*.
- Chen, K., et al. (2025). Reinforcement learning for long-horizon interactive LLM agents. arXiv.
- Trivedi, H., et al. (2024). AppWorld: A controllable world of apps and people for benchmarking interactive coding agents. *Proceedings of ACL 2024*.
