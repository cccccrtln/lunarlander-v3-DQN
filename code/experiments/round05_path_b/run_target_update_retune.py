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


def trial_cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}__300000__seed11.json"


def load_structure_only() -> dict:
    cp = trial_cache_path("trial_double_dqn_structure_only")
    return json.loads(cp.read_text(encoding="utf-8"))


def run_variant(name: str, label: str, target_update_interval: int) -> dict:
    cp = trial_cache_path(name)
    if cp.exists():
        print(f"Loading cache | {name} | budget=300000 | seed=11")
        return json.loads(cp.read_text(encoding="utf-8"))

    config = dict(DQN_BASELINE_CONFIG)
    config["target_update_interval"] = target_update_interval

    train_env = make_env(BASE_SEED)
    eval_env = make_env(BASE_SEED + 10_000)
    callback = EvalCurveCallback(eval_env=eval_env, eval_freq=CHECKPOINT_FREQ, n_eval_episodes=N_EVAL_EPISODES)

    print(f"Running experiment | {name} | budget=300000 | seed=11")
    model = DoubleDQN(seed=BASE_SEED, env=train_env, verbose=0, **config)
    model.learn(total_timesteps=LONG_BUDGET, callback=callback, progress_bar=False)

    payload = {
        "name": name,
        "label": label,
        "seed": BASE_SEED,
        "budget": LONG_BUDGET,
        "config": config,
        "records": callback.records,
    }
    cp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    train_env.close()
    eval_env.close()
    return payload


def export_plot(runs: list[dict]) -> Path:
    fig, ax = plt.subplots(figsize=(10.0, 5.4))
    for run in runs:
        df = pd.DataFrame(run["records"])
        ax.plot(df["timesteps"], df["eval_mean_return"], marker="o", linewidth=2.2, label=run["label"])
    ax.axhline(SOLVED_LINE, color="gray", linestyle="--", linewidth=1.2, label="solved threshold (200)")
    ax.set_title("Path B: Double DQN target_update_interval retuning (300k)")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Deterministic eval mean return")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best", fontsize=9)
    out = PLOT_DIR / "ddqn_retune_target_update_interval_300k.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    ensure_dirs()
    ddqn_1000 = load_structure_only()
    ddqn_1000["label"] = "Double DQN: target_update_interval=1000"
    ddqn_500 = run_variant("ddqn_tui_500", "Double DQN: target_update_interval=500", 500)
    ddqn_2000 = run_variant("ddqn_tui_2000", "Double DQN: target_update_interval=2000", 2000)
    plot_path = export_plot([ddqn_500, ddqn_1000, ddqn_2000])
    print(f"Plot written to: {plot_path}")


if __name__ == "__main__":
    main()
