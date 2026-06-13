# Report Alignment

This submission package is designed to match the content described in the final report.

## Project Overview

The code package focuses on the reported methods and the experiments that support the report's conclusions. It does **not** include every internal notebook-building utility used during research; instead, it includes the runnable scripts required to reproduce the reported baselines, path-level methods, and the integrated model.

## Mapping From Report Sections To Code

### Framework Selection: DQN vs PPO

- `experiments/round03_baselines/run_dqn_baseline.py`
- `experiments/round03_baselines/run_ppo_baseline.py`

### Path A: Tuned Vanilla DQN

- `experiments/round04_path_a/run_tuned_vanilla_dqn.py`

### Path B: Mainstream DQN Structure Variants

- `src/double_dqn.py`
- `experiments/round05_path_b/run_structure_variants.py`
- `experiments/round05_path_b/run_target_update_retune.py`
- `experiments/round05_path_b/run_update_and_clip_retune.py`
- `experiments/round05_path_b/run_double_plus_patha_combo.py`
- `experiments/round05_path_b/run_nstep_exploration_probe.py`

### External Reference Reproduction

- `references/run_github1_reference_repro.py`

### Path C: Auxiliary Heads and Objective Shaping

- `src/auxiliary_objectives.py`
- `experiments/round06_path_c/run_auxiliary_heads_baseline.py`
- `experiments/round06_path_c/run_auxiliary_feedback_probe.py`
- `experiments/round06_path_c/run_auxiliary_detached_feedback_probe.py`
- `experiments/round06_path_c/run_objective_shaping_probe.py`
- `experiments/round06_path_c/run_objective_shaping_sweep.py`

### Path D: Integrated Model

- `experiments/round07_path_d/run_integrated_probe.py`
- `experiments/round07_path_d/run_integrated_retune_updates.py`

### Plotting Current Best Models

- `experiments/analysis/plot_current_top_models.py`

## Notes

- All paths are local to the `code/` folder.
- All logs, caches, and plots are written to `code/results/`.
- Minimal placeholder notebooks for the round-03 baseline scripts are provided in `code/outputs/`.
