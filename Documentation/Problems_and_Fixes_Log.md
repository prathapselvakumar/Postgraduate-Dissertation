# Problems and Fixes Log: VCBM Dissertation and arXiv Paper

This log records every problem found in the VCBM work, from reviewing the published paper (arXiv:2610.24032v1) through re-running the experiments, rebuilding the AppWorld baseline and running the new CSF3 jobs. Each entry says what went wrong, why, what was done about it, and where it stands. Dates are 2026.

**Status key:** **Fixed** (in the code or text), **Done** (completed and verified), **Open** (still to do), **Superseded** (no longer applies).

## Summary

| # | Area | Problem | Severity | Status |
|---|---|---|---|---|
| 1 | AppWorld results | Both published AppWorld runs trained as plain LOOP; VCBM was never active | Critical | Fixed (new runs) |
| 2 | AppWorld results | Reported gains come from one training batch with different task mixes | Major | Fixed (text) |
| 3 | Watch-repair | 95% target equals the baseline's exploration ceiling | Major | Fixed (code option; documented) |
| 4 | Watch-repair | VCBM's episodes-to-converge is set by its own schedule | Major | Fixed (text) |
| 5 | Watch-repair | Result depends on a tuned setting chosen on the evaluation seeds | Major | Done (held-out seeds) |
| 6 | Watch-repair | No trajectory-level baseline; plain sample-average MC matches VCBM | Major | Done (re-run) |
| 7 | AppWorld method | Memory not persistent; fallback misdescribed; weighting jumps at k | Major | Fixed (code option; text) |
| 8 | Paper text | Overclaims, misattributed test, typos, citation errors | Minor | Fixed (v2) |
| 9 | Baseline | The "LOOP baseline" folder contained the full VCBM code path | Major | Fixed (upstream ML-LOOP folder) |
| 10 | CSF3 | Patch script failed silently; bookmark endpoint missing | Critical | Fixed |
| 11 | CSF3 | Upstream gave AppWorld 5 s to start; trial failed | Major | Fixed |
| 12 | CSF3 | Job killed at iteration 24 by an AppWorld timeout + NCCL deadlock | Critical | Fixed |
| 13 | CSF3 | Loss averaged to rank 0 only; ranks could disagree on skipping | Major | Fixed |
| 14 | CSF3 | Two jobs on one node clashed on port 29500 and could share a Ray cluster | Major | Fixed |
| 15 | CSF3 | Runs take about 3 weeks; 4-day job limit | Constraint | Fixed (auto-resubmit) |
| 16 | CSF3 | ML-LOOP segment killed at iteration 93 by the same AppWorld timeout | Critical | Fixed (resubmitted) |
| 17 | Paper | v2 watch-repair results, abstract and conclusions not yet rewritten | — | Open |
| 21 | CSF3 | Both runs died in the same minute: scratch stall, then the retry loop's own restart failed | Critical | Fixed (resubmit pending) |

---

## Part 1 — Problems found in the published paper and code

### 1. Both AppWorld runs were plain LOOP (Critical)

- **Problem.** The paper reports VCBM-blended LOOP beating plain LOOP on AppWorld (+29.4% success, −18.8% tokens). In fact neither run applied VCBM credit.
- **Evidence.** In the CSF3 job logs every `POST /execute_with_bookmark` request returned HTTP 404: all 80,332 in the baseline job (18631089) and all 103,457 in the VCBM job (18668337). The training code treats a 404 as "no bookmark", so no rollout in either run was bookmarked.
- **Cause.** `scripts/patch_appworld_server.py`, which adds the bookmark endpoint to the AppWorld server, hardcoded a `python3.12` path relative to the current folder, and when it could not find the file it printed "Error" but exited normally. Nothing noticed, and the 404 fallback was silent.
- **Also found.** The baseline's logged configuration had `vcc_blend_alpha = 1.0` (pure VCBM), not 0. It made no difference only because nothing was bookmarked. The baseline's launch script never set β.
- **Fix.**
  - The patch script now finds the `appworld` install the job will actually launch and exits with an error if it cannot patch it.
  - A pre-flight check (`scripts/preflight_bookmarks.py`) stops the job before training unless a state-changing call is bookmarked and a read-only call is not.
  - The 404 path now logs a warning.
  - Per-iteration `vcbm/*` metrics record how many rollouts had bookmarks.
  - Paper v2 and the README reframe the two runs as a measurement of run-to-run variance between two identical LOOP runs.
- **Status.** Fixed. The new VCBM run's log shows `PREFLIGHT OK` and only `200` responses (18,354 by 26 Oct).

### 2. AppWorld numbers came from one training batch (Major)

- **Problem.** Success rates were from the 48 training rollouts (12 scenarios) of the final iteration, not a held-out split. The two runs' final batches had different difficulty mixes: the baseline had 8 hard-task rollouts, VCBM 4.
- **Effect.** At the baseline's mix, VCBM's overall success would be 0.417 (+17.6%), not 0.458 (+29.4%). The +166.7% on difficulty 2 rests on about four tasks per run.
- **Fix.** Paper v2 and the README state this and use the difficulty-averaged rate. Both are now framed as a variance measurement (see item 1).
- **Status.** Fixed (text). Held-out evaluation results exist in the runs' offline W&B logs but have not been extracted.

### 3. The 95% target was the baseline's ceiling (Major)

- **Problem.** The RUDDER baseline actor explores at a constant 10% (its comment says 5%), so its best possible correct-decision rate is 0.90 + 0.10 × 0.5 = 95%, exactly the convergence target. VCBM's exploration decays to 0.5%, giving it a 99.75% ceiling.
- **Fix.** `TabularActor` takes an exploration schedule. The defaults reproduce the published 10%, and a matched schedule is available in `4. Fair-Comparison/run_fair_comparison.py`.
- **Finding.** Giving RUDDER VCBM's decaying schedule made it slower (median 5,932 episodes on the first 5 seeds), so the ceiling is a flaw in the comparison's design but not why RUDDER lost.
- **Status.** Fixed (code option). The matched RUDDER runs are incomplete (see item 17).

### 4. VCBM's convergence is set by its own schedule (Major)

- **Problem.** VCBM needs 2,000 warm-up transitions (40 episodes), then 60 forced samples per brand and action (480 episodes), then a full 750-episode window. That puts its earliest possible finish at about 1,196 episodes. 19 of 20 seeds finished between 1,196 and 1,221, which the paper presented as stability.
- **Fix.** Paper v2 Section IV-A states the floor.
- **Status.** Fixed (text).

### 5. Tuned on the evaluation seeds (Major)

- **Problem.** `forced_min_samples = 60` was chosen on seeds 1–20, the same seeds the results were reported on. Lower values make seeds fail to converge:

  | forced_min_samples | Median episodes | Seeds not converged (of 20) |
  |---|---|---|
  | 60 | 1,204 | 0 |
  | 30 | 970 | 1 |
  | 10 | 864 | 3 |
  | 5 | 3,914 | 7 |

- **Fix.** Re-ran on held-out seeds 21–40. VCBM still converged 20/20 (median 1,201.5; one outlier at 5,036), so the published number holds on new seeds.
- **Status.** Done.

### 6. Causal discovery may not be what helps (Major)

- **Problem.** The abstract claims VCBM beats "an undifferentiated trajectory-level baseline", but none was run in the tabular task. The code had MC and TD learners that were never reported.
- **Re-run results (held-out seeds 21–40):**
  - MC and TD with the constant 0.1 step size never reached 95% within 20,000 episodes.
  - **MC with sample-average values (`MC_AVG`), no causal discovery at all, was not significantly slower than VCBM:** median 1,527 vs. 1,201.5, p = 0.22.
- **Interpretation.** VCBM's advantage over RUDDER appears to come from averaging every sample and from forced balanced sampling, not from discovering the causal structure.
- **Also.** The baseline actor is hard-coded to take a fixed action after t = 0, so it is already told where the decision is. VCBM's noise marginalisation never matters here, because the noise counters are always 0 at t = 0.
- **Status.** Done (re-run). This changes the paper's central claim and must go into v2 (item 17).

### 7. AppWorld method described wrongly (Major)

- **Problems.**
  - The memory bank is rebuilt for every rollout, so it carries nothing across rollouts or iterations, despite the paper's "persistent memory" and "retrieval-based generalisation".
  - The stored return is never used.
  - With at most k bookmarks, each bookmark gets weight 1.0 and the rest α (20:1). With more than k, retrieved bookmarks get softmax weights averaging 1/k (about 3:1), so credit becomes much less concentrated exactly when a rollout has more state changes. A retrieved turn can fall below the floor.
- **Fix.**
  - Paper v2 describes the code as it actually ran.
  - A `rl.vcc_weighting_mode=unified` option always retrieves, rescales the weights to average 1, and clamps them at α. `legacy` remains the default, for reproducing the published runs.
  - The new runs use `unified`.
- **Status.** Fixed.

### 8. Paper text errors (Minor)

- **Problems.**
  - Overclaims in the abstract ("establishes the central claim", "identical configurations").
  - The Mann-Whitney test is attributed to structure recovery.
  - Table II's "Final %" is just the stopping threshold (713/750).
  - Population SD (ddof = 0) is reported as SD: 1,428.0, where the sample SD is 1,465.1.
  - The z-test has a typo (SE₀² + SE₀²).
  - `n_min = 8` is a dead parameter.
  - "Visual" is in the name, though nothing is visual.
  - Citation errors: AlphaZero cited for robotic control; saddle-point and SGD-stability papers cited for outlier gradients; an offline-RL tutorial cited for the metric; HCA given the wrong NeurIPS volume.
- **Fix.** Corrected in `Documentation/Research Paper/Arxiv/v2/`, which compiles cleanly with MiKTeX.
- **Status.** Fixed, apart from the sections in item 17.

### 9. The baseline folder was not plain LOOP (Major)

- **Problem.** `1. LOOP-Baseline` is a fork of Apple's ml-loop with the complete VCBM code path, VCBM settings in its config, and β defaulting to 1.
- **Fix.** Built `3. ML-LOOP-Upstream`: Apple's ml-loop at commit `f14107a`, plus infrastructure-only fixes needed on CSF3, with no VCBM code (0 `vcc`/bookmark references). Every change from upstream is listed in `README_CSF3.md` and `CSF3_INFRA_PATCH.diff`. Its job scripts match the VCBM ones in every non-VCBM setting.
- **Status.** Fixed. The new baseline runs from this folder.

---

## Part 2 — Problems running the experiments

### Local re-runs (watch-repair)

| Problem | Fix |
|---|---|
| `matplotlib`, `wandb` and `scipy` not installed locally | The fair-comparison runner stubs the plotting and logging imports; `analyze_results.py` computes the Mann-Whitney test without scipy |
| Constant-step MC/TD runs take hours (every seed hits the 20,000 cap) | Ran in parallel in the background |
| RUDDER per-seed numbers do not reproduce exactly (one seed failed to converge in 20,000 episodes; the paper reported 20/20) | Likely PyTorch version or thread-count nondeterminism in the LSTM; VCBM (NumPy only) reproduced exactly. To be stated in v2 |
| The session ended before the RUDDER runs finished | **Open:** re-run the missing RUDDER seeds |

### CSF3 setup

| Problem | Cause | Fix |
|---|---|---|
| `git pull` on CSF3 refused to merge | The GitHub history had been force-rewritten | Backed up, then `git reset --hard origin/main` |
| Old folder `Code/ML-Loop_vs_VCBM/` no longer tracked | Repository layout changed | Kept for its logs, W&B files and `appworld-env`; new jobs use `Code/Dissertation/...` |
| Files like `venmo_credentials.txt` in the training folders | Agent-generated code writing AppWorld's simulated data during rollouts | Confirmed synthetic AppWorld data; never committed |
| New folder had no `logs/` or `appworld-env` | Not tracked by git | `mkdir logs`; symlink to the old `appworld-env` |
| Patch script failed with `/bin/sh` syntax error | pip writes a `/bin/sh` wrapper for the `appworld` launcher when the path has spaces | Patch script reads the interpreter from the wrapper or uses the sibling `python` |
| A job script's launch command collapsed onto one line | Backslash continuations lost when generated through a shell heredoc | Regenerated from files; verified with `bash -n` |
| Two folders install a package with the same name | Both are `phi-agents` in one conda env; the last install wins | Each job puts its own folder first on `PYTHONPATH` and refuses to train if it imports another folder's code |
| Upstream ML-LOOP trial failed: "Could not connect to localhost" | Upstream waits 5 s for AppWorld; it takes 60–120 s to load on CSF3 | Ported the fork's 120 s wait (item 11) |
| `conda: command not found` on the login node | Conda is not initialised in login shells | `source ~/scratch/miniconda3/etc/profile.d/conda.sh` |
| Log files "not found" | Running commands from the other experiment's folder; old log-name prefix | Notes files list each script's log prefix and folder |

### 10–11. Startup failures (Critical / Major)

- **10.** Covered under item 1: the silent patch failure.
- **11.** Upstream's `interface.py` gave each AppWorld server 5 seconds (5 tries × 1 s) to start. Fixed by porting the fork's 60 × 2 s wait and its guard against a stuck server, without any bookmark code.

### 12. AppWorld timeout killed a 27-hour run (Critical)

- **Problem.** VCBM job 23794151 died during iteration 24 after 27 hours. One AppWorld call on GPU rank 1 exceeded the 100-second read timeout. The `requests.ReadTimeout` escaped the per-task retry loop and killed rank 1. Rank 0 had already finished its rollouts and waited in a one-number `ALLREDUCE` for 2 hours, until the NCCL watchdog aborted the job.
- **Cause.** The retry loop caught `AppWorldExecutionError`, `HTTPError` and `ConnectionError`, but `ReadTimeout` is none of these. Code execution was wrapped, but calls such as `task_completed()` and `evaluate()` were not.
- **Fix.** The retry loop catches every `requests` error (`RequestException`): the AppWorld server is restarted, which also kills agent code stuck inside it, and the task is retried. Applied to all three trainer copies.
- **Status.** Fixed. Resumed from checkpoint 23 as job 23889244.

### 13. Ranks could disagree on skipping an update (Major)

- **Problem.** Found while investigating item 12. The averaged loss was sent with `reduce(dst=0)`, so only rank 0 saw it, while each rank decided from its own copy whether to skip the optimizer step. A NaN on one rank would make one rank skip and the other enter the optimizer's collective, causing the same deadlock.
- **Fix.** `all_reduce`, in all three trainer copies.
- **Status.** Fixed. This bug is in Apple's original code; it did not cause item 12.

### 14. Two jobs sharing a node (Major)

- **Problem.** The VCBM restart (job 23830758) failed after 1 min 43 s with `EADDRINUSE` on port 29500. It landed on the same 4-GPU node as ML-LOOP. Both used PyTorch's default rendezvous port, and the VCBM health check hardcoded it.
- **Further risk.** The trainer starts Ray with `ray.init()` and no address. Ray then joins the newest cluster recorded in `/tmp/ray`, which on a shared node could be the other experiment's cluster.
- **Fix.**
  - Each job picks a free port for `accelerate` (`--main_process_port`) and the health check.
  - Each job gets its own Ray directory (`RAY_TMPDIR=/tmp/ray_<jobid>`).
- **Related.** Slurm stores a copy of a job script at submit time, so ML-LOOP's already-queued continuation lacked these fixes. It was replaced with one built from the fixed script.
- **Status.** Fixed.

### 15. Run length and the 4-day limit (Constraint)

- **Measured speed.**
  - ML-LOOP: about 62 minutes per iteration.
  - VCBM: about 74 minutes per iteration, about 20% slower, probably because of the bookmark checks.
  - At 400 iterations that is roughly 17–20 days, or 5–6 four-day segments, per arm.
- **Fix.**
  - Each job queues its own continuation, which runs only if the job hits the time limit and resumes from the last checkpoint. A finished, failed or cancelled job ends the chain (up to 10 segments).
  - A plateau-based early stop was added, then turned off by default (`CONVERGE=false`), so both arms get exactly 400 iterations. As a result, `convergence.json` is not written.
- **Decisions.** 32B on 4 GPUs was set up, then replaced by 7B on 2 GPUs per arm, so both arms run at once with the published 7B settings.

### 16. ML-LOOP killed by the same timeout (Critical)

- **Problem.** ML-LOOP's first segment (job 23794150) failed on 28 Oct at 20:23, after 3 days 10 hours, about 10 minutes after finishing iteration 93 (checkpoint 93 saved). Its log shows the same `requests.exceptions.ReadTimeout` (100-second read timeout on one AppWorld server) as item 12.
- **Cause.** The segment started before the item 12 fix and was still running the old code. The baseline was exposed to the same gap in Apple's retry loop.
- **Chaining behaved as designed.** The continuation (23889245) saw `FAILED`, logged "not continuing", and exited after 3 seconds, so a crash cannot loop.
- **Fix.** Resubmitted with the fixed code and job script; it resumes from checkpoint 93.
- **Status.** Fixed. Both arms now run the timeout-safe code.

### 21. Both runs died together; recovery itself failed (Critical)

- **Problem.** On 30 Oct both runs failed at 22:06: VCBM job 23889244 at iteration 81, ML-LOOP job 23940481 at iteration 130. Each had an AppWorld `ReadTimeout` at about 20:05, followed 2 hours later by the NCCL watchdog abort.
- **What the logs showed.**
  - The item 12 fix was working. Earlier timeouts were caught ("Caught exception while running vllm inference") on 28 and 30 Oct, and those runs recovered.
  - The fatal timeout escaped through the recovery path. The retry loop called `world.restart()` with no protection. `restart()` waits for the new server with a `requests.get` that had no timeout and caught only `ConnectionError`. A restart that failed after closing the old server left none, so the next attempt raised "Server is closed".
- **Why both at once.** Both jobs keep their AppWorld databases on the shared `~/scratch` filesystem, which was not full (47%). A stall there slows every AppWorld call on every node simultaneously, and new servers cannot load in time either.
- **Fix.** Applied identically to all three trainer copies and all four job scripts:
  - The retry loop catches any exception, protects the restart, and backs off (60 s × attempt, up to 600 s). The retry limit rose from 5 to 10.
  - `restart()` starts a fresh server if a previous restart left none.
  - The startup check has a 30 s request timeout and keeps waiting on any request error.
  - The AppWorld call timeout rose from 100 s to 300 s, so short stalls no longer trigger recovery.
  - Each job copies the AppWorld data to node-local `/tmp` when it has three times the space needed, and otherwise stays on scratch and logs that it did.
- **Status.** Fixed in the code. Both arms need resubmitting; they resume from checkpoints 81 (VCBM) and 130 (ML-LOOP).

---

## Part 3 — Still to do

17. **Paper v2.**
    - Rewrite the watch-repair results, abstract, introduction and conclusions around items 3–6, in particular that sample-average MC matches VCBM, and the RUDDER reproduction gap.
    - Finish the missing RUDDER seeds first.
    - Add the new 7B AppWorld results when both runs reach 400 iterations.
18. **Supervisor.** The submitted dissertation may contain the same AppWorld claim as arXiv v1 (item 1). Raise it before posting v2.
19. **Held-out AppWorld evaluation.** Extract the evaluator's results (every 5 iterations, 20 rollouts) from the offline W&B logs, for both the published and the new runs.
20. **Seeds.** One seed per arm is weak evidence: two identical LOOP runs differed by about 30% in success. A second seed per arm (`vcbm 2`, ML-LOOP `2`) would add more than anything else.
