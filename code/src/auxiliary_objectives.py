from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, create_mlp
from stable_baselines3.common.type_aliases import PyTorchObs
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork


@dataclass
class AuxiliaryLossWeights:
    q_td: float = 1.0
    stable: float = 0.20
    crash: float = 0.20
    timeout: float = 0.20
    abs_x: float = 0.20
    fuel: float = 0.20


@dataclass
class ObjectiveShapingWeights:
    stable: float = 0.50
    crash: float = 0.50
    timeout: float = 0.50
    abs_x: float = 0.10
    fuel: float = 0.10


class LunarLanderAuxiliaryInfoWrapper(gym.Wrapper):
    """Wrapper-like helper expected to be used around a single env before Monitor."""

    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self._main_count = 0
        self._side_count = 0

    def reset(self, **kwargs):
        self._main_count = 0
        self._side_count = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        if action == 2:
            self._main_count += 1
        elif action in (1, 3):
            self._side_count += 1

        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            info = dict(info)
            unwrapped = self.env.unwrapped
            stable_landing = int(not unwrapped.lander.awake)
            body_contact_crash = int(unwrapped.game_over)
            info["aux_targets"] = {
                "stable_landing": float(stable_landing),
                "body_contact_crash": float(body_contact_crash),
                "timeout": float(int(truncated)),
                "abs_x_T": float(abs(obs[0])),
                "fuel_cost_proxy": float(0.30 * self._main_count + 0.03 * self._side_count),
            }
        return obs, reward, terminated, truncated, info


class AuxiliaryReplayBuffer(ReplayBuffer):
    """
    Stores standard transitions only after an episode ends so terminal labels can be
    propagated back to every transition in that episode.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: th.device | str = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ) -> None:
        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination,
        )
        assert self.n_envs == 1, "AuxiliaryReplayBuffer currently supports only n_envs=1"
        self.aux_stable = np.zeros((buffer_size, self.n_envs), dtype=np.float32)
        self.aux_crash = np.zeros((buffer_size, self.n_envs), dtype=np.float32)
        self.aux_timeout = np.zeros((buffer_size, self.n_envs), dtype=np.float32)
        self.aux_abs_x = np.zeros((buffer_size, self.n_envs), dtype=np.float32)
        self.aux_fuel = np.zeros((buffer_size, self.n_envs), dtype=np.float32)
        self._pending_episode: deque[dict[str, Any]] = deque()
        self.action_dim = get_action_dim(action_space)

    def _store_with_aux(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        info: dict[str, Any],
        aux_targets: dict[str, float],
    ) -> None:
        store_pos = self.pos
        super().add(obs, next_obs, action, reward, done, [info])
        self.aux_stable[store_pos, 0] = aux_targets["stable_landing"]
        self.aux_crash[store_pos, 0] = aux_targets["body_contact_crash"]
        self.aux_timeout[store_pos, 0] = aux_targets["timeout"]
        self.aux_abs_x[store_pos, 0] = aux_targets["abs_x_T"]
        self.aux_fuel[store_pos, 0] = aux_targets["fuel_cost_proxy"]

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        # single-env assumption
        transition = {
            "obs": np.array(obs, copy=True),
            "next_obs": np.array(next_obs, copy=True),
            "action": np.array(action, copy=True).reshape((self.n_envs, self.action_dim)),
            "reward": np.array(reward, copy=True).reshape((self.n_envs,)),
            "done": np.array(done, copy=True).reshape((self.n_envs,)),
            "info": dict(infos[0]),
        }
        self._pending_episode.append(transition)

        if bool(done[0]):
            aux_targets = infos[0].get("aux_targets", None)
            if aux_targets is None:
                aux_targets = {
                    "stable_landing": 0.0,
                    "body_contact_crash": 0.0,
                    "timeout": float(infos[0].get("TimeLimit.truncated", False)),
                    "abs_x_T": 0.0,
                    "fuel_cost_proxy": 0.0,
                }
            while self._pending_episode:
                t = self._pending_episode.popleft()
                self._store_with_aux(
                    t["obs"],
                    t["next_obs"],
                    t["action"],
                    t["reward"],
                    t["done"],
                    t["info"],
                    aux_targets,
                )

    def sample(self, batch_size: int, env=None) -> dict[str, th.Tensor]:
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        env_indices = np.zeros((len(batch_inds),), dtype=np.int64)

        next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)
        obs = self._normalize_obs(self.observations[batch_inds, env_indices, :], env)
        rewards = self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env)
        dones = (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1)

        return {
            "observations": self.to_torch(obs),
            "actions": self.to_torch(self.actions[batch_inds, env_indices, :]),
            "next_observations": self.to_torch(next_obs),
            "dones": self.to_torch(dones),
            "rewards": self.to_torch(rewards),
            "stable_landing": self.to_torch(self.aux_stable[batch_inds, env_indices]),
            "body_contact_crash": self.to_torch(self.aux_crash[batch_inds, env_indices]),
            "timeout": self.to_torch(self.aux_timeout[batch_inds, env_indices]),
            "abs_x_T": self.to_torch(self.aux_abs_x[batch_inds, env_indices]),
            "fuel_cost_proxy": self.to_torch(self.aux_fuel[batch_inds, env_indices]),
        }


class AuxiliaryHeadQNetwork(QNetwork):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        net_arch: list[int] | None = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            features_dim=features_dim,
            net_arch=net_arch,
            activation_fn=activation_fn,
            normalize_images=normalize_images,
        )
        if net_arch is None:
            net_arch = [64, 64]
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.features_dim = features_dim
        self.latent_dim = net_arch[-1] if len(net_arch) > 0 else features_dim
        action_dim = int(self.action_space.n)

        trunk_modules = create_mlp(self.features_dim, -1, self.net_arch, self.activation_fn)
        self.trunk = nn.Sequential(*trunk_modules)
        self.q_head = nn.Linear(self.latent_dim, action_dim)
        self.stable_head = nn.Linear(self.latent_dim, 1)
        self.crash_head = nn.Linear(self.latent_dim, 1)
        self.timeout_head = nn.Linear(self.latent_dim, 1)
        self.abs_x_head = nn.Linear(self.latent_dim, 1)
        self.fuel_head = nn.Linear(self.latent_dim, 1)

    def _latent(self, obs: PyTorchObs) -> th.Tensor:
        features = self.extract_features(obs, self.features_extractor)
        return self.trunk(features)

    def _raw_aux_outputs(self, z: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        stable_logit = self.stable_head(z).squeeze(-1)
        crash_logit = self.crash_head(z).squeeze(-1)
        timeout_logit = self.timeout_head(z).squeeze(-1)
        abs_x = self.abs_x_head(z).squeeze(-1)
        fuel = self.fuel_head(z).squeeze(-1)
        return stable_logit, crash_logit, timeout_logit, abs_x, fuel

    def forward(self, obs: PyTorchObs) -> th.Tensor:
        z = self._latent(obs)
        return self.q_head(z)

    def auxiliary_outputs(self, obs: PyTorchObs) -> dict[str, th.Tensor]:
        z = self._latent(obs)
        stable_logit, crash_logit, timeout_logit, abs_x, fuel = self._raw_aux_outputs(z)
        return {
            "q": self.q_head(z),
            "stable_logit": stable_logit,
            "crash_logit": crash_logit,
            "timeout_logit": timeout_logit,
            "abs_x": abs_x,
            "fuel": fuel,
        }

    def _predict(self, observation: PyTorchObs, deterministic: bool = True) -> th.Tensor:
        q_values = self(observation)
        return q_values.argmax(dim=1).reshape(-1)


class AuxiliaryHeadDQNPolicy(DQNPolicy):
    def make_q_net(self) -> AuxiliaryHeadQNetwork:
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return AuxiliaryHeadQNetwork(**net_args).to(self.device)


class AuxiliaryFeedbackQNetwork(AuxiliaryHeadQNetwork):
    """
    Lets the Q head directly consume auxiliary predictions.
    The auxiliary predictions stay supervised by their own losses, while the
    fused Q head can use them as explicit decision context.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        net_arch: list[int] | None = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            features_extractor=features_extractor,
            features_dim=features_dim,
            net_arch=net_arch,
            activation_fn=activation_fn,
            normalize_images=normalize_images,
        )
        action_dim = int(self.action_space.n)
        self.feedback_head = nn.Sequential(
            nn.Linear(self.latent_dim + 5, self.latent_dim),
            self.activation_fn(),
            nn.Linear(self.latent_dim, action_dim),
        )

    def _feedback_features(
        self,
        stable_logit: th.Tensor,
        crash_logit: th.Tensor,
        timeout_logit: th.Tensor,
        abs_x: th.Tensor,
        fuel: th.Tensor,
    ) -> th.Tensor:
        # Use bounded feedback channels so the Q head sees interpretable and
        # numerically stable summaries.
        return th.stack(
            [
                th.sigmoid(stable_logit),
                th.sigmoid(crash_logit),
                th.sigmoid(timeout_logit),
                th.tanh(abs_x),
                th.tanh(fuel),
            ],
            dim=1,
        )

    def forward(self, obs: PyTorchObs) -> th.Tensor:
        return self.auxiliary_outputs(obs)["q"]

    def auxiliary_outputs(self, obs: PyTorchObs) -> dict[str, th.Tensor]:
        z = self._latent(obs)
        stable_logit, crash_logit, timeout_logit, abs_x, fuel = self._raw_aux_outputs(z)
        feedback_features = self._feedback_features(stable_logit, crash_logit, timeout_logit, abs_x, fuel)
        q_input = th.cat([z, feedback_features], dim=1)
        return {
            "q": self.feedback_head(q_input),
            "stable_logit": stable_logit,
            "crash_logit": crash_logit,
            "timeout_logit": timeout_logit,
            "abs_x": abs_x,
            "fuel": fuel,
            "feedback_features": feedback_features,
        }


class AuxiliaryFeedbackDQNPolicy(DQNPolicy):
    def make_q_net(self) -> AuxiliaryFeedbackQNetwork:
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return AuxiliaryFeedbackQNetwork(**net_args).to(self.device)


class DetachedAuxiliaryFeedbackQNetwork(AuxiliaryFeedbackQNetwork):
    """
    Reuses auxiliary predictions as Q-head context, but detaches them before fusion
    so TD updates do not directly reshape auxiliary predictors.
    """

    def auxiliary_outputs(self, obs: PyTorchObs) -> dict[str, th.Tensor]:
        z = self._latent(obs)
        stable_logit, crash_logit, timeout_logit, abs_x, fuel = self._raw_aux_outputs(z)
        feedback_features = self._feedback_features(stable_logit, crash_logit, timeout_logit, abs_x, fuel).detach()
        q_input = th.cat([z, feedback_features], dim=1)
        return {
            "q": self.feedback_head(q_input),
            "stable_logit": stable_logit,
            "crash_logit": crash_logit,
            "timeout_logit": timeout_logit,
            "abs_x": abs_x,
            "fuel": fuel,
            "feedback_features": feedback_features,
        }


class DetachedAuxiliaryFeedbackDQNPolicy(DQNPolicy):
    def make_q_net(self) -> DetachedAuxiliaryFeedbackQNetwork:
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return DetachedAuxiliaryFeedbackQNetwork(**net_args).to(self.device)


def objective_shaping_score(
    outputs: dict[str, th.Tensor],
    weights: ObjectiveShapingWeights,
) -> th.Tensor:
    stable = th.sigmoid(outputs["stable_logit"])
    crash = th.sigmoid(outputs["crash_logit"])
    timeout = th.sigmoid(outputs["timeout_logit"])
    abs_x = th.tanh(outputs["abs_x"])
    fuel = th.tanh(outputs["fuel"])
    return (
        weights.stable * stable
        - weights.crash * crash
        - weights.timeout * timeout
        - weights.abs_x * abs_x
        - weights.fuel * fuel
    )


class AuxiliaryHeadDQN(DQN):
    def __init__(
        self,
        policy="MlpPolicy",
        *args,
        auxiliary_loss_weights: AuxiliaryLossWeights | None = None,
        **kwargs,
    ):
        if policy == "MlpPolicy":
            policy = AuxiliaryHeadDQNPolicy
        if "replay_buffer_class" not in kwargs or kwargs["replay_buffer_class"] is None:
            kwargs["replay_buffer_class"] = AuxiliaryReplayBuffer
        super().__init__(policy, *args, **kwargs)
        self.auxiliary_loss_weights = auxiliary_loss_weights or AuxiliaryLossWeights()

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses: list[float] = []
        q_losses: list[float] = []
        aux_losses: list[float] = []

        current_size = self.replay_buffer.buffer_size if self.replay_buffer.full else self.replay_buffer.pos
        if current_size < batch_size:
            return

        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = replay_data.get("discounts", None)
            discounts = discounts if discounts is not None else self.gamma

            with th.no_grad():
                next_q_values = self.q_net_target(replay_data["next_observations"])
                next_q_values, _ = next_q_values.max(dim=1)
                next_q_values = next_q_values.reshape(-1, 1)
                target_q_values = replay_data["rewards"] + (1 - replay_data["dones"]) * discounts * next_q_values

            outputs = self.q_net.auxiliary_outputs(replay_data["observations"])
            current_q_values = th.gather(outputs["q"], dim=1, index=replay_data["actions"].long())
            q_loss = F.smooth_l1_loss(current_q_values, target_q_values)

            stable_loss = F.binary_cross_entropy_with_logits(outputs["stable_logit"], replay_data["stable_landing"])
            crash_loss = F.binary_cross_entropy_with_logits(outputs["crash_logit"], replay_data["body_contact_crash"])
            timeout_loss = F.binary_cross_entropy_with_logits(outputs["timeout_logit"], replay_data["timeout"])
            abs_x_loss = F.smooth_l1_loss(outputs["abs_x"], replay_data["abs_x_T"])
            fuel_loss = F.smooth_l1_loss(outputs["fuel"], replay_data["fuel_cost_proxy"])

            aux_total = (
                self.auxiliary_loss_weights.stable * stable_loss
                + self.auxiliary_loss_weights.crash * crash_loss
                + self.auxiliary_loss_weights.timeout * timeout_loss
                + self.auxiliary_loss_weights.abs_x * abs_x_loss
                + self.auxiliary_loss_weights.fuel * fuel_loss
            )
            loss = self.auxiliary_loss_weights.q_td * q_loss + aux_total

            losses.append(loss.item())
            q_losses.append(q_loss.item())
            aux_losses.append(aux_total.item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
        self.logger.record("train/q_loss", np.mean(q_losses))
        self.logger.record("train/aux_loss", np.mean(aux_losses))


class AuxiliaryFeedbackDQN(AuxiliaryHeadDQN):
    def __init__(
        self,
        policy="MlpPolicy",
        *args,
        auxiliary_loss_weights: AuxiliaryLossWeights | None = None,
        **kwargs,
    ):
        if policy == "MlpPolicy":
            policy = AuxiliaryFeedbackDQNPolicy
        super().__init__(
            policy,
            *args,
            auxiliary_loss_weights=auxiliary_loss_weights,
            **kwargs,
        )


class DetachedAuxiliaryFeedbackDQN(AuxiliaryHeadDQN):
    def __init__(
        self,
        policy="MlpPolicy",
        *args,
        auxiliary_loss_weights: AuxiliaryLossWeights | None = None,
        **kwargs,
    ):
        if policy == "MlpPolicy":
            policy = DetachedAuxiliaryFeedbackDQNPolicy
        super().__init__(
            policy,
            *args,
            auxiliary_loss_weights=auxiliary_loss_weights,
            **kwargs,
        )


class AuxiliaryObjectiveShapingDQN(AuxiliaryHeadDQN):
    def __init__(
        self,
        policy="MlpPolicy",
        *args,
        auxiliary_loss_weights: AuxiliaryLossWeights | None = None,
        objective_shaping_weights: ObjectiveShapingWeights | None = None,
        **kwargs,
    ):
        super().__init__(
            policy,
            *args,
            auxiliary_loss_weights=auxiliary_loss_weights,
            **kwargs,
        )
        self.objective_shaping_weights = objective_shaping_weights or ObjectiveShapingWeights()

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses: list[float] = []
        q_losses: list[float] = []
        aux_losses: list[float] = []
        shaping_scores: list[float] = []

        current_size = self.replay_buffer.buffer_size if self.replay_buffer.full else self.replay_buffer.pos
        if current_size < batch_size:
            return

        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = replay_data.get("discounts", None)
            discounts = discounts if discounts is not None else self.gamma

            with th.no_grad():
                next_outputs = self.q_net_target.auxiliary_outputs(replay_data["next_observations"])
                next_q_values = next_outputs["q"].max(dim=1, keepdim=True)[0]
                shaped_bonus = objective_shaping_score(next_outputs, self.objective_shaping_weights).reshape(-1, 1)
                target_q_values = replay_data["rewards"] + (1 - replay_data["dones"]) * discounts * (next_q_values + shaped_bonus)

            outputs = self.q_net.auxiliary_outputs(replay_data["observations"])
            current_q_values = th.gather(outputs["q"], dim=1, index=replay_data["actions"].long())
            q_loss = F.smooth_l1_loss(current_q_values, target_q_values)

            stable_loss = F.binary_cross_entropy_with_logits(outputs["stable_logit"], replay_data["stable_landing"])
            crash_loss = F.binary_cross_entropy_with_logits(outputs["crash_logit"], replay_data["body_contact_crash"])
            timeout_loss = F.binary_cross_entropy_with_logits(outputs["timeout_logit"], replay_data["timeout"])
            abs_x_loss = F.smooth_l1_loss(outputs["abs_x"], replay_data["abs_x_T"])
            fuel_loss = F.smooth_l1_loss(outputs["fuel"], replay_data["fuel_cost_proxy"])

            aux_total = (
                self.auxiliary_loss_weights.stable * stable_loss
                + self.auxiliary_loss_weights.crash * crash_loss
                + self.auxiliary_loss_weights.timeout * timeout_loss
                + self.auxiliary_loss_weights.abs_x * abs_x_loss
                + self.auxiliary_loss_weights.fuel * fuel_loss
            )
            loss = self.auxiliary_loss_weights.q_td * q_loss + aux_total

            losses.append(loss.item())
            q_losses.append(q_loss.item())
            aux_losses.append(aux_total.item())
            shaping_scores.append(shaped_bonus.mean().item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
        self.logger.record("train/q_loss", np.mean(q_losses))
        self.logger.record("train/aux_loss", np.mean(aux_losses))
        self.logger.record("train/shaping_score", np.mean(shaping_scores))


class DoubleAuxiliaryObjectiveShapingDQN(AuxiliaryHeadDQN):
    """
    Combines Path B's Double-DQN target calculation with Path C's five-head
    objective shaping, while allowing Path A tuned hyperparameters to be passed
    in through the normal DQN config.
    """

    def __init__(
        self,
        policy="MlpPolicy",
        *args,
        auxiliary_loss_weights: AuxiliaryLossWeights | None = None,
        objective_shaping_weights: ObjectiveShapingWeights | None = None,
        **kwargs,
    ):
        super().__init__(
            policy,
            *args,
            auxiliary_loss_weights=auxiliary_loss_weights,
            **kwargs,
        )
        self.objective_shaping_weights = objective_shaping_weights or ObjectiveShapingWeights()

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses: list[float] = []
        q_losses: list[float] = []
        aux_losses: list[float] = []
        shaping_scores: list[float] = []

        current_size = self.replay_buffer.buffer_size if self.replay_buffer.full else self.replay_buffer.pos
        if current_size < batch_size:
            return

        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = replay_data.get("discounts", None)
            discounts = discounts if discounts is not None else self.gamma

            with th.no_grad():
                next_outputs_online = self.q_net.auxiliary_outputs(replay_data["next_observations"])
                next_actions = next_outputs_online["q"].argmax(dim=1, keepdim=True)

                next_outputs_target = self.q_net_target.auxiliary_outputs(replay_data["next_observations"])
                next_q_values = th.gather(next_outputs_target["q"], dim=1, index=next_actions.long())
                shaped_bonus = objective_shaping_score(next_outputs_target, self.objective_shaping_weights).reshape(-1, 1)
                target_q_values = replay_data["rewards"] + (1 - replay_data["dones"]) * discounts * (next_q_values + shaped_bonus)

            outputs = self.q_net.auxiliary_outputs(replay_data["observations"])
            current_q_values = th.gather(outputs["q"], dim=1, index=replay_data["actions"].long())
            q_loss = F.smooth_l1_loss(current_q_values, target_q_values)

            stable_loss = F.binary_cross_entropy_with_logits(outputs["stable_logit"], replay_data["stable_landing"])
            crash_loss = F.binary_cross_entropy_with_logits(outputs["crash_logit"], replay_data["body_contact_crash"])
            timeout_loss = F.binary_cross_entropy_with_logits(outputs["timeout_logit"], replay_data["timeout"])
            abs_x_loss = F.smooth_l1_loss(outputs["abs_x"], replay_data["abs_x_T"])
            fuel_loss = F.smooth_l1_loss(outputs["fuel"], replay_data["fuel_cost_proxy"])

            aux_total = (
                self.auxiliary_loss_weights.stable * stable_loss
                + self.auxiliary_loss_weights.crash * crash_loss
                + self.auxiliary_loss_weights.timeout * timeout_loss
                + self.auxiliary_loss_weights.abs_x * abs_x_loss
                + self.auxiliary_loss_weights.fuel * fuel_loss
            )
            loss = self.auxiliary_loss_weights.q_td * q_loss + aux_total

            losses.append(loss.item())
            q_losses.append(q_loss.item())
            aux_losses.append(aux_total.item())
            shaping_scores.append(shaped_bonus.mean().item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
        self.logger.record("train/q_loss", np.mean(q_losses))
        self.logger.record("train/aux_loss", np.mean(aux_losses))
        self.logger.record("train/shaping_score", np.mean(shaping_scores))
