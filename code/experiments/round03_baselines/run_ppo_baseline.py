from __future__ import annotations

import json
import math
import shutil
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = PROJECT_ROOT / "outputs" / "round_03_framework_baseline_comparison" / "ppo.ipynb"
TEMP_ROOT = PROJECT_ROOT / "results" / "round_03_framework_baseline_comparison" / "_tmp_ppo_run"
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

PPO_BASELINE_CONFIG = {
    "policy": "MlpPolicy",
    "n_steps": 1024,
    "batch_size": 256,
    "n_epochs": 10,
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "policy_kwargs": {
        "net_arch": [128, 128],
    },
}

N_ENVS = 8


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
    mean_max_prob_proxy: float
    approx_kl_last: float
    clip_fraction_last: float
    entropy_last: float
    explained_variance_last: float


class PPOEvalAndLogCallback(BaseCallback):
    def __init__(self, eval_env: gym.Env, eval_freq: int, n_eval_episodes: int):
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.eval_timesteps = []
        self.eval_means = []
        self.eval_stds = []
        self.approx_kl = []
        self.clip_fraction = []
        self.entropy_loss = []
        self.explained_variance = []

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
                    obs, reward, done, truncated, info = self.eval_env.step(action)
                    total_reward += float(reward)
                returns.append(total_reward)

            self.eval_timesteps.append(self.num_timesteps)
            self.eval_means.append(float(np.mean(returns)))
            self.eval_stds.append(float(np.std(returns)))

            logged = getattr(self.model.logger, "name_to_value", {})
            self.approx_kl.append(float(logged.get("train/approx_kl", math.nan)))
            self.clip_fraction.append(float(logged.get("train/clip_fraction", math.nan)))
            self.entropy_loss.append(float(logged.get("train/entropy_loss", math.nan)))
            self.explained_variance.append(float(logged.get("train/explained_variance", math.nan)))

            print(
                f"[eval] steps={self.num_timesteps} "
                f"mean_return={self.eval_means[-1]:.2f} std={self.eval_stds[-1]:.2f} "
                f"kl={self.approx_kl[-1]:.4f} clip={self.clip_fraction[-1]:.4f}"
            )
        return True


def make_train_env(seed: int):
    return make_vec_env(ENV_ID, n_envs=N_ENVS, seed=seed)


def make_eval_env(seed: int) -> gym.Env:
    env = gym.make(ENV_ID)
    env.reset(seed=seed)
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    return Monitor(env)


def deterministic_rollout_metrics(model: PPO, seed: int, n_episodes: int = 10):
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


def representative_action_probs(model: PPO, states: np.ndarray):
    obs_tensor, _ = model.policy.obs_to_tensor(states)
    with th.no_grad():
        dist = model.policy.get_distribution(obs_tensor)
        probs = dist.distribution.probs.cpu().numpy()
    return pd.DataFrame(probs, columns=["action_0", "action_1", "action_2", "action_3"])


def compute_steps_to_threshold(eval_timesteps, eval_means, threshold=THRESHOLD):
    for t, m in zip(eval_timesteps, eval_means):
        if m >= threshold:
            return int(t)
    return None


def train_one_run(budget_label: str, seed: int) -> tuple[RunSummary, dict]:
    total_timesteps = BUDGET_TO_TIMESTEPS[budget_label]
    train_env = make_train_env(seed)
    eval_env = make_eval_env(seed + 10_000)
    callback = PPOEvalAndLogCallback(eval_env, EVAL_FREQ, N_EVAL_EPISODES)
    model = PPO(seed=seed, env=train_env, verbose=0, **PPO_BASELINE_CONFIG)

    start = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=False)
    elapsed = time.time() - start

    rollout_df = deterministic_rollout_metrics(model, seed=seed + 50_000, n_episodes=10)
    prob_df = representative_action_probs(model, REPRESENTATIVE_STATES)
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
        mean_max_prob_proxy=float(prob_df.max(axis=1).mean()),
        approx_kl_last=float(callback.approx_kl[-1]) if callback.approx_kl else math.nan,
        clip_fraction_last=float(callback.clip_fraction[-1]) if callback.clip_fraction else math.nan,
        entropy_last=float(callback.entropy_loss[-1]) if callback.entropy_loss else math.nan,
        explained_variance_last=float(callback.explained_variance[-1]) if callback.explained_variance else math.nan,
    )

    artifacts = {
        "eval_timesteps": callback.eval_timesteps,
        "eval_means": callback.eval_means,
        "eval_stds": callback.eval_stds,
        "approx_kl": callback.approx_kl,
        "clip_fraction": callback.clip_fraction,
        "entropy_loss": callback.entropy_loss,
        "explained_variance": callback.explained_variance,
        "rollout_df": rollout_df,
        "prob_df": prob_df,
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
        mean_kl = df["approx_kl_last"].mean()
        mean_clip = df["clip_fraction_last"].mean()
        parts.append(f"### {budget}")
        parts.append(f"- Mean final evaluation return: {mean_return:.2f}")
        parts.append(f"- Mean success proxy: {mean_success:.3f}")
        parts.append(f"- Mean crash proxy: {mean_crash:.3f}")
        parts.append(f"- Mean approx_kl: {mean_kl:.4f}")
        parts.append(f"- Mean clip_fraction: {mean_clip:.4f}")
        if mean_success < 0.30 and mean_return > 0:
            parts.append("- Preliminary view: process-level behavior is improving, but full safe landing remains rare. The bottleneck is more likely rollout efficiency or advantage signals not yet translating into terminal quality.")
        elif mean_kl < 0.005 and mean_clip < 0.05:
            parts.append("- Preliminary view: updates may be too conservative, limiting policy movement and slowing down take-off in fixed budgets.")
        else:
            parts.append("- Preliminary view: the baseline is forming a global control style, but it still needs finer diagnosis using key-state action distributions and failure modes.")
        parts.append("")
    return "\n".join(parts)


def update_notebook(summary_df: pd.DataFrame, results_by_budget: dict):
    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    summary_md = [
        "## Executed Results (auto-generated)",
        "",
        "### Primary metrics for each run",
        "",
        summary_df.to_markdown(index=False),
        "",
        plot_budget_curves(results_by_budget),
        "",
        make_bottleneck_diagnosis(summary_df),
        "",
        "## Suggested Next Steps (auto draft)",
        "",
        "1. First check whether rollout length and sample reuse leave on-policy data under-utilized.",
        "2. Then check whether advantage estimation and value-baseline quality are limiting terminal quality.",
        "3. Finally check whether the update step is too conservative for the fixed budget.",
        "",
    ]

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
            print(f"Running PPO baseline | budget={budget_label} | seed={seed}")
            summary, artifacts = train_one_run(budget_label, seed)
            summaries.append(asdict(summary))
            results_by_budget[budget_label].append({"summary": summary, "artifacts": artifacts})

    summary_df = pd.DataFrame(summaries)
    update_notebook(summary_df, results_by_budget)
    print("PPO baseline finished. Results were written back to the notebook.")

    if TEMP_ROOT.exists():
        shutil.rmtree(TEMP_ROOT)


if __name__ == "__main__":
    main()
