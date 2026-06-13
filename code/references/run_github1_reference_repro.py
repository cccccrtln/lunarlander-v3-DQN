from __future__ import annotations

import json
import random
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from stable_baselines3.common.monitor import Monitor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "round_05_github1_ddqn_repro"
CACHE_DIR = RESULTS_DIR / "_cache"
PLOT_DIR = RESULTS_DIR / "plots"

ENV_ID = "LunarLander-v3"
BASE_SEED = 11
TOTAL_TIMESTEPS = 300_000
CHECKPOINT_FREQ = 10_000
N_EVAL_EPISODES = 10
SOLVED_LINE = 200.0


@dataclass
class Github1DDQNConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    buffer_size: int = 100_000
    batch_size: int = 256
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_decay: float = 0.998
    epsilon_min: float = 0.1
    tau: float = 0.001
    grad_clip: float = 1.0
    n_step: int = 3
    warmup_episodes: int = 300
    warmup_reset_epsilon: float = 0.95


class Github1DQNNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=0.1)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Github1DDQNAgent:
    def __init__(self, obs_dim: int, action_dim: int, config: Github1DDQNConfig, device: torch.device):
        self.device = device
        self.config = config
        self.action_dim = action_dim

        self.q_net = Github1DQNNet(obs_dim, action_dim).to(device)
        self.target_net = Github1DQNNet(obs_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = torch.optim.Adam(
            self.q_net.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.memory: deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=config.buffer_size)
        self.n_step_buffer: deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=config.n_step)
        self.epsilon = config.epsilon_start

    def act(self, state: np.ndarray, greedy: bool = False) -> int:
        if (not greedy) and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return int(torch.argmax(q_values).item())

    def _calculate_n_step_return(self) -> tuple[np.ndarray, int, float, np.ndarray, bool]:
        n_step_reward = 0.0
        for i, (_, _, reward, _, done) in enumerate(self.n_step_buffer):
            n_step_reward += (self.config.gamma**i) * float(reward)
            if done:
                break

        first_state, first_action, _, _, _ = self.n_step_buffer[0]
        _, _, _, last_next_state, last_done = self.n_step_buffer[-1]
        return first_state, first_action, n_step_reward, last_next_state, last_done

    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.n_step_buffer.append((state, action, reward, next_state, done))
        if len(self.n_step_buffer) == self.config.n_step:
            self.memory.append(self._calculate_n_step_return())
        if done:
            self.n_step_buffer.clear()

    def replay(self) -> tuple[float | None, float | None]:
        if len(self.memory) < self.config.batch_size:
            return None, None

        batch = random.sample(self.memory, self.config.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.as_tensor(np.array(states), dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(np.array(actions), dtype=torch.long, device=self.device)
        rewards_t = torch.as_tensor(np.array(rewards), dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(np.array(dones), dtype=torch.float32, device=self.device)

        rewards_t = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-7)

        current_q = self.q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.q_net(next_states_t).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states_t).gather(1, next_actions).squeeze(1)
            target_q = rewards_t + (1 - dones_t) * (self.config.gamma ** self.config.n_step) * next_q

        loss = nn.SmoothL1Loss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.config.grad_clip)
        self.optimizer.step()

        for target_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)

        self.epsilon = max(self.config.epsilon_min, self.epsilon * self.config.epsilon_decay)
        return float(loss.item()), float(grad_norm.item())


def ensure_dirs() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)


def make_env(seed: int) -> Monitor:
    env = gym.make(ENV_ID)
    env.reset(seed=seed)
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    return Monitor(env)


def cache_path(total_timesteps: int, seed: int) -> Path:
    return CACHE_DIR / f"github1_ddqn_repro__{total_timesteps}__seed{seed}.json"


def evaluate(agent: Github1DDQNAgent, n_episodes: int, base_seed: int) -> dict:
    env = gym.make(ENV_ID)
    returns = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=base_seed + ep)
        done = False
        truncated = False
        total_reward = 0.0
        while not (done or truncated):
            action = agent.act(obs, greedy=True)
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += float(reward)
        returns.append(total_reward)
    env.close()
    return {
        "eval_mean_return": float(np.mean(returns)),
        "eval_std_return": float(np.std(returns)),
    }


def run_repro(total_timesteps: int, seed: int) -> dict:
    ensure_dirs()
    cp = cache_path(total_timesteps, seed)
    if cp.exists():
        print(f"Loading cache | github1_ddqn_repro | budget={total_timesteps} | seed={seed}")
        return json.loads(cp.read_text(encoding="utf-8"))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = make_env(seed)
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(env.action_space.n)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Github1DDQNConfig()
    agent = Github1DDQNAgent(obs_dim, action_dim, config, device)

    records: list[dict] = []
    obs, _ = env.reset(seed=seed)
    total_steps = 0
    episode_idx = 0

    while total_steps < total_timesteps:
        if episode_idx < config.warmup_episodes:
            agent.epsilon = 1.0
        elif episode_idx == config.warmup_episodes:
            agent.epsilon = config.warmup_reset_epsilon

        episode_done = False
        while not episode_done and total_steps < total_timesteps:
            action = agent.act(obs, greedy=False)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            agent.remember(obs, action, reward, next_obs, done)
            agent.replay()

            total_steps += 1
            obs = next_obs
            episode_done = done

            if total_steps % CHECKPOINT_FREQ == 0:
                eval_stats = evaluate(agent, N_EVAL_EPISODES, base_seed=50_000 + total_steps)
                records.append(
                    {
                        "timesteps": total_steps,
                        "eval_mean_return": eval_stats["eval_mean_return"],
                        "eval_std_return": eval_stats["eval_std_return"],
                    }
                )
                print(
                    f"[eval] steps={total_steps} mean_return={eval_stats['eval_mean_return']:.2f} "
                    f"std={eval_stats['eval_std_return']:.2f}"
                )

        episode_idx += 1
        obs, _ = env.reset(seed=seed + episode_idx)

    env.close()

    payload = {
        "label": "trial: github1 DDQN repro (300k)",
        "seed": seed,
        "budget": total_timesteps,
        "config": asdict(config),
        "records": records,
    }
    cp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def export_plot(control_run: dict, repro_run: dict) -> Path:
    fig, ax = plt.subplots(figsize=(10.0, 5.4))
    for run, linestyle in [(control_run, "--"), (repro_run, "-")]:
        df = pd.DataFrame(run["records"])
        ax.plot(
            df["timesteps"],
            df["eval_mean_return"],
            marker="o",
            linewidth=2.2,
            linestyle=linestyle,
            label=run["label"],
        )
    ax.axhline(SOLVED_LINE, color="gray", linestyle="--", linewidth=1.2, label="solved threshold (200)")
    ax.set_title("github1 DDQN repro vs Path A combo C (300k)")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Deterministic eval mean return")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best", fontsize=9)
    out = PLOT_DIR / "github1_repro_vs_pathA_comboC_300k.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def load_path_a_control() -> dict:
    cp = (
        PROJECT_ROOT
        / "results"
        / "round_04_path_a_vanilla_dqn_tuning"
        / "_single_factor_screening"
        / "cache"
        / "combo__combo_aggressive_64__300000.json"
    )
    payload = json.loads(cp.read_text(encoding="utf-8"))
    return {
        "label": "control: Path A combo C (300k)",
        "records": payload["records"],
        "config": payload["config"],
    }


def main() -> None:
    control_run = load_path_a_control()
    repro_run = run_repro(TOTAL_TIMESTEPS, BASE_SEED)
    plot_path = export_plot(control_run, repro_run)
    print(f"Plot written to: {plot_path}")


if __name__ == "__main__":
    main()
