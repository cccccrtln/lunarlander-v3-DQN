from __future__ import annotations

import json
import math
import shutil
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as th
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = PROJECT_ROOT / "outputs" / "round_03_framework_baseline_comparison" / "dqn.ipynb"
TEMP_ROOT = PROJECT_ROOT / "results" / "round_03_framework_baseline_comparison" / "_tmp_dqn_run"
ENV_ID = "LunarLander-v3"
EVAL_FREQ = 10_000
N_EVAL_EPISODES = 10
THRESHOLD = 200.0

BUDGET_TO_SEEDS = {
    "100k": [11, 29],
    "300k": [11, 29, 47],
}

BUDGET_TO_TIMESTEPS = {
    "100k": 100_000,
    "300k": 300_000,
}

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
    "exploration_fraction": 0.15,
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.05,
    "policy_kwargs": {
        "net_arch": [128, 128],
    },
}


@dataclass
class RunSummary:
    budget: str
    seed: int
    auc: float
    return_last_eval: float
    eval_std_last: float
    steps_to_threshold: int | None
    success_rate_proxy: float
    crash_rate_proxy: float
    timeout_rate_proxy: float
    main_engine_usage_proxy: float
    side_engine_usage_proxy: float
    final_abs_vx: float
    final_abs_vy: float
    final_abs_angle: float
    mean_max_q_proxy: float
    final_exploration_rate: float
    episode_count_seen: int


class DQNEvalAndLogCallback(BaseCallback):
    def __init__(self, eval_env: gym.Env, eval_freq: int, n_eval_episodes: int):
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.eval_timesteps = []
        self.eval_means = []
        self.eval_stds = []
        self.exploration_rates = []
        self.episode_returns = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for info, done in zip(infos, dones):
            if done and "episode" in info:
                ep = info["episode"]
                self.episode_returns.append(float(ep["r"]))
                self.episode_lengths.append(float(ep["l"]))

        if self.eval_freq > 0 and self.num_timesteps % self.eval_freq == 0:
            returns = []
            for ep in range(self.n_eval_episodes):
                obs, _ = self.eval_env.reset(seed=10_000 + self.num_timesteps + ep)
                done = False
                truncated = False
                total_reward = 0.0
                while not (done or truncated):
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, info = self.eval_env.step(action)
                    total_reward += float(reward)
                returns.append(total_reward)

            self.eval_timesteps.append(self.num_timesteps)
            self.eval_means.append(float(np.mean(returns)))
            self.eval_stds.append(float(np.std(returns)))
            self.exploration_rates.append(float(getattr(self.model, "exploration_rate", math.nan)))
            print(
                f"[eval] steps={self.num_timesteps} "
                f"mean_return={self.eval_means[-1]:.2f} std={self.eval_stds[-1]:.2f} "
                f"epsilon={self.exploration_rates[-1]:.4f}"
            )
        return True


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_env(seed: int, mode: str) -> gym.Env:
    env = gym.make(ENV_ID)
    env.reset(seed=seed)
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    return Monitor(env)


def deterministic_rollout_metrics(model: DQN, seed: int, n_episodes: int = 10):
    env = gym.make(ENV_ID)
    metrics = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        truncated = False
        total_reward = 0.0
        length = 0
        action_hist = {0: 0, 1: 0, 2: 0, 3: 0}
        last_obs = None
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            action_hist[action] += 1
            obs, reward, done, truncated, info = env.step(action)
            total_reward += float(reward)
            length += 1
            last_obs = obs
        vx = float(last_obs[2]) if last_obs is not None else math.nan
        vy = float(last_obs[3]) if last_obs is not None else math.nan
        angle = float(last_obs[4]) if last_obs is not None else math.nan
        left_contact = int(last_obs[6]) if last_obs is not None else 0
        right_contact = int(last_obs[7]) if last_obs is not None else 0
        success = int(left_contact == 1 and right_contact == 1 and abs(vx) < 0.2 and abs(vy) < 0.2 and abs(angle) < 0.2)
        crash = int(done and success == 0 and abs(vy) >= 0.2)
        timeout = int(truncated)
        metrics.append(
            {
                "episode_return": total_reward,
                "episode_length": length,
                "landing_success": success,
                "crash": crash,
                "timeout": timeout,
                "main_engine_usage_proxy": action_hist.get(2, 0),
                "side_engine_usage_proxy": action_hist.get(1, 0) + action_hist.get(3, 0),
                "final_abs_vx": abs(vx),
                "final_abs_vy": abs(vy),
                "final_abs_angle": abs(angle),
            }
        )
    env.close()
    return pd.DataFrame(metrics)


REPRESENTATIVE_STATES = np.array(
    [
        [0.0, 1.2, 0.0, -0.8, 0.0, 0.0, 0.0, 0.0],
        [0.4, 0.9, 0.3, -0.6, 0.1, 0.1, 0.0, 0.0],
        [-0.3, 0.4, -0.4, -0.3, -0.2, 0.2, 0.0, 0.0],
        [0.1, 0.2, 0.1, -0.1, 0.05, 0.0, 0.0, 0.0],
    ],
    dtype=np.float32,
)
REPRESENTATIVE_STATE_LABELS = [
    "high_altitude_fast_descent",
    "drifting_with_tilt",
    "low_altitude_lateral_error",
    "final_approach_small_correction",
]


def representative_state_q_values(model: DQN, states: np.ndarray):
    with th.no_grad():
        tensor = th.as_tensor(states, dtype=th.float32, device=model.device)
        q_values = model.q_net(tensor).cpu().numpy()
    return pd.DataFrame(q_values, columns=["action_0", "action_1", "action_2", "action_3"])


def compute_steps_to_threshold(eval_timesteps, eval_means, threshold=THRESHOLD):
    for t, m in zip(eval_timesteps, eval_means):
        if m >= threshold:
            return int(t)
    return None


def train_one_run(budget_label: str, seed: int) -> tuple[RunSummary, dict]:
    total_timesteps = BUDGET_TO_TIMESTEPS[budget_label]
    train_env = make_env(seed, "train")
    eval_env = make_env(seed + 10_000, "eval")

    callback = DQNEvalAndLogCallback(eval_env, EVAL_FREQ, N_EVAL_EPISODES)
    model = DQN(
        seed=seed,
        env=train_env,
        verbose=0,
        **DQN_BASELINE_CONFIG,
    )

    start = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=False)
    elapsed = time.time() - start

    rollout_df = deterministic_rollout_metrics(model, seed=seed + 50_000, n_episodes=10)
    q_df = representative_state_q_values(model, REPRESENTATIVE_STATES)
    auc = float(np.trapz(callback.eval_means, callback.eval_timesteps) / max(float(total_timesteps), 1.0))
    summary = RunSummary(
        budget=budget_label,
        seed=seed,
        auc=auc,
        return_last_eval=float(callback.eval_means[-1]) if callback.eval_means else math.nan,
        eval_std_last=float(callback.eval_stds[-1]) if callback.eval_stds else math.nan,
        steps_to_threshold=compute_steps_to_threshold(callback.eval_timesteps, callback.eval_means),
        success_rate_proxy=float(rollout_df["landing_success"].mean()),
        crash_rate_proxy=float(rollout_df["crash"].mean()),
        timeout_rate_proxy=float(rollout_df["timeout"].mean()),
        main_engine_usage_proxy=float(rollout_df["main_engine_usage_proxy"].mean()),
        side_engine_usage_proxy=float(rollout_df["side_engine_usage_proxy"].mean()),
        final_abs_vx=float(rollout_df["final_abs_vx"].mean()),
        final_abs_vy=float(rollout_df["final_abs_vy"].mean()),
        final_abs_angle=float(rollout_df["final_abs_angle"].mean()),
        mean_max_q_proxy=float(q_df.max(axis=1).mean()),
        final_exploration_rate=float(getattr(model, "exploration_rate", math.nan)),
        episode_count_seen=len(callback.episode_returns),
    )

    artifacts = {
        "eval_timesteps": callback.eval_timesteps,
        "eval_means": callback.eval_means,
        "eval_stds": callback.eval_stds,
        "exploration_rates": callback.exploration_rates,
        "episode_returns": callback.episode_returns,
        "episode_lengths": callback.episode_lengths,
        "rollout_df": rollout_df,
        "q_df": q_df,
        "elapsed_seconds": elapsed,
    }

    train_env.close()
    eval_env.close()
    return summary, artifacts


def plot_budget_curves(results_by_budget):
    out = []
    for budget, runs in results_by_budget.items():
        lines = [f"### {budget} evaluation curve summary"]
        for run in runs:
            seed = run["summary"].seed
            final_ret = run["summary"].return_last_eval
            auc = run["summary"].auc
            lines.append(f"- seed {seed}: final evaluation return {final_ret:.2f}, normalized AUC {auc:.4f}")
        out.append("\n".join(lines))
    return "\n\n".join(out)


def make_bottleneck_diagnosis(summary_df: pd.DataFrame) -> str:
    parts = ["## Preliminary Bottleneck Diagnosis (auto-generated)", ""]
    for budget, df in summary_df.groupby("budget"):
        mean_return = df["return_last_eval"].mean()
        mean_success = df["success_rate_proxy"].mean()
        mean_crash = df["crash_rate_proxy"].mean()
        mean_vy = df["final_abs_vy"].mean()
        parts.append(f"### {budget}")
        parts.append(f"- Mean final evaluation return: {mean_return:.2f}")
        parts.append(f"- Mean success proxy: {mean_success:.3f}")
        parts.append(f"- Mean crash proxy: {mean_crash:.3f}")
        parts.append(f"- Mean final vertical-speed magnitude: {mean_vy:.3f}")
        if mean_success < 0.30 and mean_return > 0:
            parts.append("- Preliminary view: local control is improving, but full safe landing is still limited. The bottleneck is more likely a lack of high-quality trajectories or stronger local reward exploitation than global coordination.")
        elif mean_crash > 0.50 and mean_vy > 0.25:
            parts.append("- Preliminary view: crashes remain frequent and the final vertical speed is still large. The bottleneck is more likely exploration quality or insufficient braking behavior.")
        else:
            parts.append("- Preliminary view: the baseline has learned some control capability, but it still requires finer diagnosis using key-state Q ordering and failure modes.")
        parts.append("")
    return "\n".join(parts)


def update_notebook(summary_df: pd.DataFrame, results_by_budget: dict):
    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    summary_md = ["## Executed Results (auto-generated)", "", "### Primary metrics for each run", "", summary_df.to_markdown(index=False), "", plot_budget_curves(results_by_budget), "", make_bottleneck_diagnosis(summary_df), "", "## Suggested Next Steps (auto draft)", "", "1. First check whether exploration delays the arrival of high-quality trajectories.", "2. Then check whether replay reuse is too weak or is amplifying bias.", "3. The next high-priority candidate remains value-estimation-related improvements.", ""]

    inserted = False
    for cell in nb["cells"]:
        if cell["cell_type"] == "markdown" and "## Next-step Improvement Ideas" in "".join(cell.get("source", [])):
            cell["source"] = [line + "\n" for line in "\n".join(summary_md).splitlines()]
            inserted = True
            break
    if not inserted:
        nb["cells"].append(
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [line + "\n" for line in "\n".join(summary_md).splitlines()],
            }
        )
    NOTEBOOK_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    if TEMP_ROOT.exists():
        shutil.rmtree(TEMP_ROOT)
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)

    summaries = []
    results_by_budget = {}
    for budget_label in BUDGET_TO_TIMESTEPS:
        results_by_budget[budget_label] = []
        for seed in BUDGET_TO_SEEDS[budget_label]:
            print(f"Running DQN baseline | budget={budget_label} | seed={seed}")
            summary, artifacts = train_one_run(budget_label, seed)
            summaries.append(asdict(summary))
            results_by_budget[budget_label].append({"summary": summary, "artifacts": artifacts})

    summary_df = pd.DataFrame(summaries)
    update_notebook(summary_df, results_by_budget)
    print("DQN baseline finished. Results were written back to the notebook.")

    if TEMP_ROOT.exists():
        shutil.rmtree(TEMP_ROOT)


if __name__ == "__main__":
    main()
