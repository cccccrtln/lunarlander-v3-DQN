# LunarLander RL Submission Code

This folder is the clean submission package for the LunarLander project. It is intended to be submitted together with the PDF report. The package is organized around the structure of the report and contains the runnable code needed to reproduce the reported baselines, path-level methods, and the integrated model.

## 1. Project Overview

The project studies reinforcement learning on `LunarLander-v3` under fixed interaction budgets (`100k` and `300k`). The goal is not only to train an agent that lands successfully, but also to understand:

1. whether `DQN` or `PPO` is the better research backbone for this task,
2. what kinds of improvements help `DQN` the most,
3. and how far those improvements can be combined.

The report is built around four stages:

- baseline comparison (`DQN` vs `PPO`),
- `Path A`: tuned vanilla `DQN`,
- `Path B`: mainstream structure variants,
- `Path C`: auxiliary heads and objective shaping,
- `Path D`: integration of the best ideas.

## 2. Environment

All experiments target:

- `LunarLander-v3`
- discrete action space
- default environment reward
- fixed evaluation protocol based on deterministic evaluation every `10k` environment steps

## 3. Agent Actions and Reward Logic

The environment uses four discrete actions:

- `0`: do nothing
- `1`: fire left orientation engine
- `2`: fire main engine
- `3`: fire right orientation engine

The environment reward can be conceptually viewed as:

`r_t ≈ terminal_reward + progress_feedback + leg_contact_reward - engine_cost`

This is why the project evaluates not only return, but also stability, crash behavior, timeout behavior, final lateral offset, and fuel-cost proxy.

## 4. Final Methods Covered In This Package

### Framework Selection

- `DQN` baseline
- `PPO` baseline

### Path A: Tuned Vanilla DQN

The final reported tuned vanilla DQN uses:

- `learning_rate=1e-4`
- `gamma=0.99`
- `target_update_interval=1000`
- `learning_starts=10000`
- `net_arch=[128, 128]`
- `train_freq=1`
- `gradient_steps=4`
- `max_grad_norm=1.0`
- `batch_size=64`

### Path B: Structure Variants

This package includes the code used to evaluate:

- `Double DQN`
- `Dueling Double DQN`
- `Double DQN + n-step(3)`
- `Double DQN + Path A tuned parameters`

### Path C: Auxiliary Heads and Objective Shaping

The package includes the five auxiliary targets used in the report:

- `P(stable_landing | s)`
- `P(body_contact_crash | s)`
- `P(timeout | s)`
- `E[abs(x_T) | s]`
- `E[fuel_cost_proxy | s]`

It also includes:

- plain auxiliary-head supervision,
- direct auxiliary-to-Q feedback,
- detached auxiliary-to-Q feedback,
- five-head objective shaping,
- and coefficient sweep scripts for objective shaping.

### Path D: Integrated Model

The integrated model combines:

- `Path A` tuned update configuration,
- `Path B`'s `Double DQN`,
- `Path C`'s five-head objective shaping.

The package also includes the first round of joint retuning for the integrated model.

## 5. Folder Structure

```text
code/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── auxiliary_objectives.py
│   └── double_dqn.py
├── experiments/
│   ├── round03_baselines/
│   ├── round04_path_a/
│   ├── round05_path_b/
│   ├── round06_path_c/
│   ├── round07_path_d/
│   └── analysis/
├── references/
├── outputs/
├── results/
└── docs/
```

## 6. How To Reproduce The Main Results

Install dependencies first:

```bash
pip install -r requirements.txt
```

Then run the experiments you need.

### Baselines

```bash
python experiments/round03_baselines/run_dqn_baseline.py
python experiments/round03_baselines/run_ppo_baseline.py
```

### Path A

```bash
python experiments/round04_path_a/run_tuned_vanilla_dqn.py
```

### Path B

```bash
python experiments/round05_path_b/run_structure_variants.py
python experiments/round05_path_b/run_target_update_retune.py
python experiments/round05_path_b/run_update_and_clip_retune.py
python experiments/round05_path_b/run_double_plus_patha_combo.py
python experiments/round05_path_b/run_nstep_exploration_probe.py
```

### Path C

```bash
python experiments/round06_path_c/run_auxiliary_heads_baseline.py
python experiments/round06_path_c/run_auxiliary_feedback_probe.py
python experiments/round06_path_c/run_auxiliary_detached_feedback_probe.py
python experiments/round06_path_c/run_objective_shaping_probe.py
python experiments/round06_path_c/run_objective_shaping_sweep.py
```

### Path D

```bash
python experiments/round07_path_d/run_integrated_probe.py
python experiments/round07_path_d/run_integrated_retune_updates.py
```

### Current Top Models Plot

```bash
python experiments/analysis/plot_current_top_models.py
```

## 7. Results, Outputs, and Deliverables

- Generated caches and plots are written under `results/`.
- Generated notebooks or presentation artifacts, when used, are written under `outputs/`.
- `docs/report_alignment.md` explains how the code package maps to the report sections.

## 8. Notes On Packaging

This package is intentionally cleaner than the full research workspace:

- internal one-off utilities and early notebook-construction helpers are omitted,
- only the methods and experiment runners directly relevant to the report are included,
- all packaging docs are written in English,
- and all scripts are made local to the `code/` folder so the package can stand alone as a submission artifact.
