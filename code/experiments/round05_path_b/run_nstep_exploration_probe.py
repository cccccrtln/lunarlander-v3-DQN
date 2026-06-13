from __future__ import annotations

import json
import sys
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src"))

from double_dqn import DoubleDQN


ENV_ID = "LunarLander-v3"
CHECKPOINT_FREQ = 10_000
N_EVAL_EPISODES = 10
BASE_SEED = 11
LONG_BUDGET = 300_000
SOLVED_LINE = 200.0

RESULTS_DIR = PROJECT_ROOT / "results" / "round_05_path_b_dqn_structure_variants"
CACHE_DIR = RESULTS_DIR / "_cache"
PLOT_DIR = RESULTS_DIR / "plots"

DQN_BASELINE_CONFIG = {
    "policy": "MlpPolicy",
    "learning_rate": 1e-4,
    "buffer_size": 100_000,
    "learning_starts": 10_000,
    "batch_size": 64,
    "train_freq": 4,
    "gradient_steps": 1,
    "gamma": 0.99,
    "target_update_interval": 1000,
    "max_grad_norm": 10.0,
    "exploration_fraction": 0.15,
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.05,
    "policy_kwargs": {"net_arch": [128, 128]},
}


class EvalCurveCallback(BaseCallback):
    def __init__(self, eval_env: gym.Env, eval_freq: int, n_eval_episodes: int):
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.records: list[dict] = []

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.num_timesteps % self.eval_freq == 0:
            returns = []
            for ep in range(self.n_eval_episodes):
                obs, _ = self.eval_env.reset(seed=20_000 + self.num_timesteps + ep)
                done = False
                truncated = False
                total_reward = 0.0
                while not (done or truncated):
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, _ = self.eval_env.step(action)
                    total_reward += float(reward)
                returns.append(total_reward)

            self.records.append(
                {
                    "timesteps": self.num_timesteps,
                    "eval_mean_return": float(np.mean(returns)),
                    "eval_std_return": float(np.std(returns)),
                }
            )
            print(
                f"[eval] steps={self.num_timesteps} "
                f"mean_return={self.records[-1]['eval_mean_return']:.2f} "
                f"std={self.records[-1]['eval_std_return']:.2f}"
            )
        return True


def ensure_dirs() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)


def make_env(seed: int) -> gym.Env:
    env = gym.make(ENV_ID)
    env.reset(seed=seed)
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    return Monitor(env)


def load_nstep_baseline() -> dict:
    cp = CACHE_DIR / "trial_double_dqn_nstep3_structure_only__300000__seed11.json"
    return json.loads(cp.read_text(encoding="utf-8"))


def probe_cache_path() -> Path:
    return CACHE_DIR / "trial_double_dqn_nstep3_exploration_probe__300000__seed11.json"


def run_probe() -> dict:
    cp = probe_cache_path()
    if cp.exists():
        print("Loading cache | trial_double_dqn_nstep3_exploration_probe | budget=300000 | seed=11")
        return json.loads(cp.read_text(encoding="utf-8"))

    config = dict(DQN_BASELINE_CONFIG)
    config["n_steps"] = 3
    config["exploration_fraction"] = 0.30
    config["exploration_final_eps"] = 0.10

    train_env = make_env(BASE_SEED)
    eval_env = make_env(BASE_SEED + 10_000)
    callback = EvalCurveCallback(eval_env=eval_env, eval_freq=CHECKPOINT_FREQ, n_eval_episodes=N_EVAL_EPISODES)

    print("Running experiment | trial_double_dqn_nstep3_exploration_probe | budget=300000 | seed=11")
    model = DoubleDQN(seed=BASE_SEED, env=train_env, verbose=0, **config)
    model.learn(total_timesteps=LONG_BUDGET, callback=callback, progress_bar=False)

    payload = {
        "name": "trial_double_dqn_nstep3_exploration_probe",
        "label": "trial: Double DQN + n-step(3) + stronger exploration",
        "seed": BASE_SEED,
        "budget": LONG_BUDGET,
        "config": config,
        "records": callback.records,
    }
    cp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    train_env.close()
    eval_env.close()
    return payload


def export_plot(baseline_run: dict, probe_run: dict) -> Path:
    fig, ax = plt.subplots(figsize=(10.0, 5.4))
    for run, linestyle in [(baseline_run, "--"), (probe_run, "-")]:
        df = pd.DataFrame(run["records"])
        ax.plot(df["timesteps"], df["eval_mean_return"], marker="o", linewidth=2.3, linestyle=linestyle, label=run["label"])
    ax.axhline(SOLVED_LINE, color="gray", linestyle="--", linewidth=1.2, label="solved threshold (200)")
    ax.set_title("Path B probe: Double+n-step baseline vs stronger-exploration variant (300k)")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Deterministic eval mean return")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best", fontsize=9)
    out = PLOT_DIR / "ddqn_nstep_exploration_probe_300k.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    ensure_dirs()
    baseline_run = load_nstep_baseline()
    baseline_run["label"] = "baseline: Double DQN + n-step(3)"
    probe_run = run_probe()
    plot_path = export_plot(baseline_run, probe_run)
    print(f"Plot written to: {plot_path}")


if __name__ == "__main__":
    main()
