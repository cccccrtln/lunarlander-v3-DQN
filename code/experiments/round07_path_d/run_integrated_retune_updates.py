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

from auxiliary_objectives import (
    AuxiliaryLossWeights,
    DoubleAuxiliaryObjectiveShapingDQN,
    LunarLanderAuxiliaryInfoWrapper,
    ObjectiveShapingWeights,
)


ENV_ID = "LunarLander-v3"
CHECKPOINT_FREQ = 10_000
N_EVAL_EPISODES = 10
BASE_SEED = 11
LONG_BUDGET = 300_000
SOLVED_LINE = 200.0

BASE_INTEGRATED_CACHE = (
    PROJECT_ROOT
    / "results"
    / "round_07_path_d_integrated_dqn"
    / "_cache"
    / "trial_pathD_integrated__300000__seed11.json"
)

RESULTS_DIR = PROJECT_ROOT / "results" / "round_07_path_d_integrated_dqn"
CACHE_DIR = RESULTS_DIR / "_cache"
PLOT_DIR = RESULTS_DIR / "plots"

BASE_CONFIG = {
    "policy": "MlpPolicy",
    "learning_rate": 1e-4,
    "buffer_size": 100_000,
    "learning_starts": 10_000,
    "batch_size": 64,
    "train_freq": 1,
    "gradient_steps": 4,
    "gamma": 0.99,
    "target_update_interval": 1000,
    "max_grad_norm": 1.0,
    "exploration_fraction": 0.15,
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.05,
    "policy_kwargs": {"net_arch": [128, 128]},
}

AUX_WEIGHTS_015 = AuxiliaryLossWeights(
    q_td=1.0,
    stable=0.15,
    crash=0.15,
    timeout=0.15,
    abs_x=0.15,
    fuel=0.15,
)

OBJECTIVE_BASE = ObjectiveShapingWeights(
    stable=0.50,
    crash=0.50,
    timeout=0.50,
    abs_x=0.10,
    fuel=0.10,
)

RETUNE_CONFIGS = {
    "base": {
        "label": "Path D base (tf=1, gs=4, clip=1.0)",
        "config": BASE_CONFIG,
    },
    "mid_updates": {
        "label": "Path D mid updates (tf=1, gs=2, clip=1.0)",
        "config": {
            **BASE_CONFIG,
            "gradient_steps": 2,
        },
    },
    "conservative_updates": {
        "label": "Path D conservative updates (tf=4, gs=1, clip=1.0)",
        "config": {
            **BASE_CONFIG,
            "train_freq": 4,
            "gradient_steps": 1,
        },
    },
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


def make_train_env(seed: int) -> gym.Env:
    env = gym.make(ENV_ID)
    env = LunarLanderAuxiliaryInfoWrapper(env)
    env.reset(seed=seed)
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    return Monitor(env)


def make_eval_env(seed: int) -> gym.Env:
    env = gym.make(ENV_ID)
    env.reset(seed=seed)
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    return Monitor(env)


def load_json_run(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def trial_cache_path(tag: str) -> Path:
    if tag == "base":
        return BASE_INTEGRATED_CACHE
    return CACHE_DIR / f"trial_pathD_{tag}__300000__seed11.json"


def run_trial(tag: str, label: str, config: dict) -> dict:
    cp = trial_cache_path(tag)
    if cp.exists():
        print(f"Loading cache | path_d_{tag} | budget=300000 | seed=11")
        return json.loads(cp.read_text(encoding="utf-8"))

    train_env = make_train_env(BASE_SEED)
    eval_env = make_eval_env(BASE_SEED + 10_000)
    callback = EvalCurveCallback(eval_env=eval_env, eval_freq=CHECKPOINT_FREQ, n_eval_episodes=N_EVAL_EPISODES)

    print(f"Running experiment | path_d_{tag} | budget=300000 | seed=11")
    model = DoubleAuxiliaryObjectiveShapingDQN(
        seed=BASE_SEED,
        env=train_env,
        verbose=0,
        auxiliary_loss_weights=AUX_WEIGHTS_015,
        objective_shaping_weights=OBJECTIVE_BASE,
        **config,
    )
    model.learn(total_timesteps=LONG_BUDGET, callback=callback, progress_bar=False)

    payload = {
        "name": f"trial_pathD_{tag}",
        "label": label,
        "seed": BASE_SEED,
        "budget": LONG_BUDGET,
        "config": config,
        "aux_weights": {
            "q_td": AUX_WEIGHTS_015.q_td,
            "stable": AUX_WEIGHTS_015.stable,
            "crash": AUX_WEIGHTS_015.crash,
            "timeout": AUX_WEIGHTS_015.timeout,
            "abs_x": AUX_WEIGHTS_015.abs_x,
            "fuel": AUX_WEIGHTS_015.fuel,
        },
        "shaping_weights": {
            "stable": OBJECTIVE_BASE.stable,
            "crash": OBJECTIVE_BASE.crash,
            "timeout": OBJECTIVE_BASE.timeout,
            "abs_x": OBJECTIVE_BASE.abs_x,
            "fuel": OBJECTIVE_BASE.fuel,
        },
        "records": callback.records,
    }
    cp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    train_env.close()
    eval_env.close()
    return payload


def export_plot(base_run: dict, mid_run: dict, conservative_run: dict) -> Path:
    fig, ax = plt.subplots(figsize=(10.6, 5.8))
    runs = [
        (base_run, "--"),
        (mid_run, "-."),
        (conservative_run, "-"),
    ]
    for run, linestyle in runs:
        df = pd.DataFrame(run["records"])
        ax.plot(df["timesteps"], df["eval_mean_return"], marker="o", linewidth=2.2, linestyle=linestyle, label=run["label"])
    ax.axhline(SOLVED_LINE, color="gray", linestyle="--", linewidth=1.2, label="solved threshold (200)")
    ax.set_title("Round 07 Path D: update-intensity retuning (300k)")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Deterministic eval mean return")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best", fontsize=9)
    out = PLOT_DIR / "path_d_retune_update_intensity_300k.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    ensure_dirs()
    base_run = run_trial("base", RETUNE_CONFIGS["base"]["label"], RETUNE_CONFIGS["base"]["config"])
    mid_run = run_trial("mid_updates", RETUNE_CONFIGS["mid_updates"]["label"], RETUNE_CONFIGS["mid_updates"]["config"])
    conservative_run = run_trial(
        "conservative_updates",
        RETUNE_CONFIGS["conservative_updates"]["label"],
        RETUNE_CONFIGS["conservative_updates"]["config"],
    )
    plot_path = export_plot(base_run, mid_run, conservative_run)
    print(f"Plot written to: {plot_path}")


if __name__ == "__main__":
    main()
