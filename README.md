# CS4486 LunarLander Project

This repository contains the working materials for a CS4486 reinforcement learning project on `LunarLander-v3`.

The main submission-ready code package is under [`code/`](./code), which includes:

- experiment runners for the baseline and final DQN study,
- the reusable source modules in `code/src/`,
- and packaging documentation aligned to the report in `code/docs/`.

Other top-level folders are kept for project organization:

- `report/` stores report deliverables and drafting material,
- `agents/` stores multi-agent workflow notes and logs,
- `results/` and `artifacts/` are local experiment outputs and are ignored from Git by default.

## Recommended GitHub Scope

This repository is configured to track source code, documentation, and report files, while excluding regenerated outputs such as caches, checkpoints, large archives, videos, and local experiment results.

## Quick Start

To run the submission package locally:

```bash
cd code
pip install -r requirements.txt
python experiments/round03_baselines/run_dqn_baseline.py
```

For the full experiment list, see [`code/README.md`](./code/README.md).
