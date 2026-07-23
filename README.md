# Postgraduate Dissertation 🎓

**Visual Causal Chains for Long-Horizon Credit Assignment in Reinforcement Learning**

> **⚠️ Status:** Work in progress. MSc Robotics, University of Manchester (graduating September 2026).

## Overview

This repository contains the codebase, research materials, and LaTeX source files for my
postgraduate dissertation. The research focuses on advanced reinforcement learning (RL)
concepts, specifically:

- Visual causal chains and event detection
- Credit assignment in long-horizon environments and sparse-reward settings
- Vision-based event memory and episodic memory
- Reward redistribution and return decomposition

The proposed approach — **Visual Causal Chains (VCC)** — identifies structurally significant
state transitions in a trajectory and uses them as anchors for credit assignment, in contrast
to purely temporal methods such as RUDDER or leave-one-out advantage estimators such as LOOP.

## Repository Structure

```
Postgraduate-Dissertation/
├── Code/                     Experiment codebases (see Code/README.md)
│   ├── Rudder/                Toy-task baseline: RUDDER vs. VCC on watch-repair
│   └── ML-Loop/                Main experiments: LOOP baseline vs. VCC on AppWorld
├── Documentation/             Dissertation process documents, in chronological order
├── Reference Documents/       Literature review data (Semantic Scholar scrape)
└── Tex/                       LaTeX sources for proposals and reports
```

### [Code/](Code/)

All experimental code. See [Code/README.md](Code/README.md) for details on each subproject.
Two research tracks are maintained in parallel:

| Track | Purpose | Baseline | VCC variant |
|---|---|---|---|
| `Code/Rudder/` | Small-scale proof of concept (pocket-watch repair task) | `1. RUDDER-Baseline` | `2. VCC-Experiment` |
| `Code/ML-Loop/` | Main-line experiments on the [AppWorld](https://appworld.dev/) benchmark, forked from [apple/ml-loop](https://github.com/apple/ml-loop) | `1. LOOP-Baseline` | `2. VCC-Experiment` |

### [Documentation/](Documentation/)

Numbered chronologically to track dissertation progress from proposal to submission:

| Folder | Contents |
|---|---|
| `0. Risk Assessment` | Working-from-home risk assessment |
| `1. Dissertation Topic Proposal` | Initial topic proposal |
| `2. First Supervisor Meeting` | Notes from first supervisor meeting |
| `3. Final Submission Documents` | Detailed report and final submission materials |
| `4. Research Plan` | Publication roadmap: contribution, experiment plan, target venues |

### [Reference Documents/](Reference%20Documents/)

Literature review dataset scraped via the Semantic Scholar API (see `fetch.py` in the research
plan). See [Reference Documents/README.md](Reference%20Documents/README.md).

### [Tex/](Tex/)

LaTeX sources for the dissertation topic proposal and final proposal.

## Current Progress

- **Literature Review Pipeline:** Automated Python data collection script (`fetch.py`) using the
  Semantic Scholar API to scrape, filter, and aggregate relevant academic papers.
- **Baseline Prototyping:** RUDDER vs. VCC comparison implemented on a toy delayed-reward task
  (pocket-watch repair) to validate the causal credit-assignment mechanism at small scale.
- **Main-line Experiments:** LOOP baseline vs. VCC comparison on the AppWorld benchmark, running
  on the University of Manchester CSF3 cluster.
- **Documentation Setup:** LaTeX build environment with automated PDF compilation for drafting
  dissertation proposals and the final document.

*(More details and methodology will be added as the research progresses.)*
