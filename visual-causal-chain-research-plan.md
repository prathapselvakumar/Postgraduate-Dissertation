# Visual Causal Chains for Long-Horizon Credit Assignment
## Research Publication Roadmap — Brutally Honest Edition

**Author:** Prathap Selvakumar
**Affiliation:** MSc Robotics, University of Manchester (graduating Sept 2026)
**Target:** Workshop paper → NeurIPS/ICML 2026 or ICLR 2027 full paper
**Date:** June 2026

---

## 1. The Real Problem You Are Solving

### 1.1 What is the problem?

In long-horizon reinforcement learning, a reward signal arrives at the **end** of a trajectory that may span 40–100+ steps. The agent must infer which early actions caused the final outcome — this is the **credit assignment problem**.

In apple/ml-loop's LOOP algorithm (AppWorld benchmark):
- Trajectories span up to 40 API interactions
- The terminal reward (0 or 1) must be distributed back over all policy tokens
- LOOP uses a **leave-one-out Monte Carlo advantage estimator** — no value network, no temporal difference
- The advantage signal exists in `policy_token_info.log_probs` but is **never visualized or inspected** per turn

The `html_utils.py` wandb logger renders raw chat messages only. Researchers cannot see which of 40 interactions drove the credit signal. This is not just a tooling gap — it means researchers are **flying blind** when diagnosing why the agent fails on long chains.

### 1.2 What is the specific gap in the literature?

| Method | Credit Mechanism | Visual Interpretability | Long-horizon? |
|--------|-----------------|------------------------|---------------|
| RUDDER | LSTM return decomposition | ✗ | ✓ |
| HCA (Hindsight Credit Assignment) | State-conditional policy | ✗ | Partial |
| LOOP (Apple, 2025) | Leave-one-out MC advantage | ✗ | ✓ |
| GAE (Schulman et al.) | TD(λ) advantage | ✗ | Partial |
| **Visual Causal Chains (proposed)** | Causal event detection + memory retrieval | **✓** | ✓ |

**The gap:** No existing long-horizon credit assignment method provides a visual, inspectable causal chain that links early actions to terminal rewards. Visualization is treated as post-hoc tooling, not as a core architectural component.

### 1.3 Why does this matter for publication?

A reviewer will ask: *"Is this a research contribution or just a dashboard?"*

The answer must be: **the causal chain detection itself is the contribution** — and it produces better credit assignment signals than LOOP/RUDDER, measurably. The visualization is the interpretability layer that makes the mechanism inspectable, not the contribution itself.

---

## 2. The Proposed Contribution

### 2.1 Core claim (must be falsifiable)

> **Visual Causal Chain credit assignment improves sample efficiency and return in sparse-reward long-horizon tasks compared to temporal difference and leave-one-out baselines, and produces causal chains that are human-verifiable.**

This claim requires:
1. An implemented 4-component architecture
2. Baseline comparisons (at minimum LOOP, RUDDER, standard PPO)
3. Evaluation on at least one long-horizon environment (MiniGrid-KeyCorridorS6 or similar)
4. Human verification study (even small-scale: 5–10 annotators, Amazon MTurk, or classmates)

### 2.2 The 4-component architecture (your dissertation design)

```
┌─────────────────────────────────────────────────────────────────┐
│                  VISUAL CAUSAL CHAIN SYSTEM                     │
├──────────────────┬──────────────────┬──────────────────────────┤
│   COMPONENT 1    │   COMPONENT 2    │      COMPONENT 3         │
│   Encoder        │  Event Detector  │     Memory Bank          │
│                  │                  │                          │
│  CNN / ViT       │  Δ-state change  │  Episodic key-value      │
│  obs → z_t       │  detector        │  store: (z_t, a_t, r_t)  │
│                  │  flags causal    │  keyed by causal events  │
│                  │  events          │                          │
├──────────────────┴──────────────────┴──────────────────────────┤
│                       COMPONENT 4                               │
│                   Retrieval + Credit                            │
│                                                                 │
│  At reward time: retrieve top-k causal events from memory       │
│  bank via cosine similarity → assign credit proportionally      │
│  → update policy on retrieved causal turns, not all turns       │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 What makes this different from RUDDER?

RUDDER uses an LSTM to decompose returns backwards through time — it has no notion of **which state changes were causally significant**. It redistributes reward over all time steps weighted by temporal contribution.

Your approach uses a **causal event detector** that identifies structurally significant state transitions (e.g., picking up a key, opening a door) and uses these as **anchors** for credit. This is fundamentally different: RUDDER redistributes credit temporally; VCC distributes credit causally.

### 2.4 Connection to apple/ml-loop

In the LOOP setting, "visual causal events" map to API call outcomes that change persistent state:
- `transfer_funds()` → balance changes → causal event
- `send_sms()` → message delivered → causal event
- `get_contact()` → no state change → not a causal anchor

The event detector applied to LOOP trajectories would identify which API calls constitute causal bookmarks, then assign LOOP's advantage signal preferentially to those turns rather than uniformly across all output tokens (`pg_per_token` in `rl/conf/rl/params/default.yaml`).

---

## 3. What You Actually Need to Build

### 3.1 Minimum viable implementation for a workshop paper

```
Priority 1 — Must have:
  ✓ Working VCC agent in MiniGrid-KeyCorridorS3 (at minimum)
  ✓ PPO baseline (Stable-Baselines3, 5 seeds)
  ✓ RUDDER baseline (reference implementation exists)
  ✓ Learning curves: return vs environment steps
  ✓ Causal chain visualization (your React tool, or matplotlib)

Priority 2 — Strong paper:
  ✓ MiniGrid-KeyCorridorS6 (longer horizon, harder)
  ✓ HCA baseline
  ✓ Ablation: no event detector (random anchors)
  ✓ Ablation: no memory bank (recency-weighted credit)
  ✓ Human verification study (even N=5)

Priority 3 — Full conference paper:
  ✓ AppWorld / LOOP integration
  ✓ Scaling to 32B LLM agent
  ✓ Statistical significance tests (Welch's t-test, effect sizes)
  ✓ Transfer study (does causal chain learned in S3 transfer to S6?)
```

### 3.2 What AI can build for you (and what it cannot)

**AI can build:**
- MiniGrid environment wrappers and observation encoders
- PPO training loop (boilerplate)
- RUDDER LSTM return decomposition baseline
- Memory bank (key-value store with cosine retrieval — this is ~50 lines of PyTorch)
- The React visualization tool (already done)
- LaTeX paper draft with your experimental results filled in
- The `html_utils.py` patch for apple/ml-loop

**AI cannot provide:**
- The insight of *why* causal events should be anchors (that's your intellectual contribution)
- The judgment of *which ablations matter* (you need to understand the system)
- Defense of results under reviewer questioning (you must understand every number)
- The human study design and its validity

**The honest timeline:**

```
June 2026       — Implement encoder + event detector in MiniGrid (AI-assisted)
July 2026       — Run experiments, PPO + RUDDER baselines (2–3 weeks GPU time)
August 2026     — Dissertation submission + preliminary results
September 2026  — Workshop paper draft (NeurIPS workshops deadline ~Sept 2026)
Oct–Dec 2026    — Refine for ICLR 2027 (deadline Oct 2026)
```

---

## 4. The Publishing Landscape — Where to Submit

### 4.1 Realistic targets in order of difficulty

| Venue | Deadline | Acceptance Rate | Fit | Verdict |
|-------|----------|-----------------|-----|---------|
| NeurIPS 2026 Workshop (XAI/RL) | ~Sept 2026 | ~40% | ✓✓ | **Primary target** |
| ICML 2026 Workshop | Passed | — | ✓✓ | Next year |
| ICLR 2027 (full paper) | Oct 2026 | ~30% | ✓ | Stretch goal |
| AAAI 2027 | Aug 2026 | ~20% | ✓ | Possible |
| IEEE RA-L (robotics angle) | Rolling | ~30% | Partial | If you pivot to robot tasks |
| IGI Global (your existing outlet) | Rolling | High | ✓ | Safe fallback |

### 4.2 What a workshop paper needs (minimum bar)

- 4–8 pages
- One environment, two baselines, learning curves
- One ablation (remove one component, show it hurts)
- One visualization figure that is genuinely novel
- A clear falsifiable claim in the abstract

This is achievable with your timeline **if** you start implementation now.

### 4.3 What reviewers will actually ask

1. *"How does event detection generalize? Does it work without hand-crafted state change thresholds?"*
   → You need a learned detector, not a rule-based one, or a strong argument for rules.

2. *"What is the computational overhead of the memory bank retrieval during training?"*
   → You need to report this. If it's slow, it kills the paper.

3. *"Why MiniGrid? This is a toy environment."*
   → Frame it as controlled evaluation: you can verify causal chains ground-truth in MiniGrid. AppWorld is the application.

4. *"RUDDER already addresses long-horizon credit. What does causality add?"*
   → The event detector selects *which* steps receive credit, not just *how much*. Different mechanism, different failure modes.

5. *"Is the human verification study statistically valid with N=5?"*
   → Be honest: "preliminary human evaluation, to be expanded." N=5 is fine for a workshop.

---

## 5. The AI-Assisted Implementation Plan

### 5.1 Week-by-week (June–August 2026)

**Week 1–2 (June 6–20): Environment + Encoder**
```python
# Ask Claude Code / Cursor to build:
- MiniGrid-KeyCorridorS3 wrapper with RGB observation
- CNN encoder: obs → 128-dim z_t
- State change detector: |z_t - z_{t-1}| > threshold → causal event flag
- PPO baseline using Stable-Baselines3
```

**Week 3–4 (June 20 – July 4): Memory Bank + Retrieval**
```python
# Memory bank structure:
memory = {
    "keys":   torch.Tensor,  # z_t for causal events only
    "values": torch.Tensor,  # (action, reward_to_go) pairs
    "steps":  list[int],     # timestep indices
}

# Retrieval at reward time:
def retrieve_causal_credits(z_terminal, memory, top_k=5):
    similarities = F.cosine_similarity(z_terminal, memory["keys"])
    top_k_idx = similarities.topk(top_k).indices
    return memory["steps"][top_k_idx], similarities[top_k_idx]
```

**Week 5–6 (July 4–18): Baselines + Experiments**
```
- RUDDER: use reference implementation (github.com/ml-jku/rudder)
- Run 5 seeds each: VCC, PPO, RUDDER
- Log: episode return, sample efficiency, causal chain length
- GPU: Colab Pro or University cluster
```

**Week 7–8 (July 18 – Aug 1): Ablations + Visualization**
```
- Ablation A: random event anchors (no detector)
- Ablation B: uniform credit (no retrieval, just PPO)
- Generate causal chain visualizations for 3–5 representative episodes
- Human annotation study (5 annotators, 10 trajectories each)
```

**Week 9–10 (Aug 1–15): Dissertation write-up**
```
- Dissertation chapter = paper body (reuse ~70% of content)
- Submit dissertation Aug/Sept 2026
```

**Week 11–12 (Aug 15 – Sept): Workshop paper**
```
- Condense dissertation to 6-page workshop format
- Submit to NeurIPS 2026 workshop
```

### 5.2 GPU requirements

| Task | Estimated Time | Platform |
|------|---------------|----------|
| PPO baseline (5 seeds, MiniGrid-S3) | ~3h | Colab free |
| RUDDER baseline (5 seeds) | ~5h | Colab Pro |
| VCC full system (5 seeds) | ~8h | Colab Pro / UoM cluster |
| Ablations (2 × 5 seeds) | ~10h | UoM cluster |

**Total:** ~26 GPU hours. University of Manchester CS cluster access is free for MSc students — use it.

---

## 6. The apple/ml-loop Contribution (Bonus Track)

### 6.1 The specific fix

The open issue in apple/ml-loop is that `rollout_html_wandb()` in `phi_agents/evals/html_utils.py` renders no credit signal. The fix is:

```python
# In html_utils.py — add advantage colour coding
def get_message_html(msg, advantage=None):
    bg_color = "transparent"
    if advantage is not None:
        if advantage > 0.6:  bg_color = "rgba(34,197,94,0.15)"
        elif advantage > 0.2: bg_color = "rgba(234,179,8,0.12)"
        else:                 bg_color = "rgba(239,68,68,0.12)"
    # ... render with bg_color
```

```python
# In appworld_rollout_data.py — thread advantages through
def log_appworld_rollouts_html(rollouts, tokenizer, writer, step):
    for i, r in enumerate(rollouts[:3]):
        # Compute per-turn advantages from policy_token_info
        advantages = compute_per_turn_advantages(r.policy_token_info, r.messages)
        log_dict[f"html/rollout_{i}"] = rollout_html_wandb(r, tokenizer, advantages)
```

### 6.2 How to frame this for your paper

> *"We demonstrate our Visual Causal Chain bookmarking system applied to apple/ml-loop (Chen et al., 2025), a state-of-the-art LOOP RL agent for long-horizon interactive tasks. We implement per-turn causal credit visualization as a patch to the existing wandb logging pipeline, enabling researchers to inspect which API interactions drove the terminal reward signal — a capability absent from the original codebase."*

This positions apple/ml-loop as your **application domain**, while MiniGrid is your **controlled evaluation**. That's a strong structure for a paper.

---

## 7. Honest Risk Assessment

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| VCC does not outperform PPO baseline | Medium | Publish as negative result + analysis; still a workshop paper |
| Dissertation deadline conflicts with experiments | High | Start experiments immediately; reuse dissertation GPU runs |
| Reviewer rejects "toy environment" (MiniGrid) | Medium | Add AppWorld qualitative results even without training |
| AI-generated code has bugs you don't understand | High | Review every component line by line; run unit tests |
| Memory bank retrieval too slow for training | Low | Use FAISS; benchmark overhead in ablations |
| NeurIPS workshop deadline missed | Medium | Fall back to AAAI 2027 or ICLR 2027 |

---

## 8. The One Thing That Will Make or Break This

The difference between a publishable paper and an MSc dissertation is a **single clear result** that reviewers cannot argue with.

For this paper, that result is:

> *"VCC achieves X% higher return than PPO and Y% better sample efficiency than RUDDER on MiniGrid-KeyCorridorS6 (N=5 seeds, p<0.05), while producing causal chains that human annotators agree with Z% of the time."*

Fill in those numbers. Everything else is scaffolding.

---

## 9. Recommended Next Steps (This Week)

1. **Today:** Set up MiniGrid + SB3 PPO baseline, confirm it trains to convergence
2. **This week:** Implement the CNN encoder and Δ-state event detector (Claude Code can do this with you)
3. **Next week:** Run PPO baseline 5 seeds, save learning curves
4. **Contact:** Email Dr. Pawel Ladosz and ask if the causal chain work can count toward any dissertation extension or if there's a UoM GPU cluster allocation you can use
5. **Optional:** Open a GitHub issue or PR on apple/ml-loop with your `html_utils.py` fix — it's a real contribution, it's visible, and it gets your name on the repo

---

*This document reflects the honest gap between "interesting dissertation idea" and "publishable research." The path is real and achievable. The question is execution speed.*
