from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from src.backtest.simulator import simulate_week


class ElecEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        bundle: dict[str, Any],
        week_sequence: list[pd.Timestamp],
        config: dict[str, Any],
    ) -> None:
        super().__init__()
        if not week_sequence:
            raise ValueError("week_sequence 为空，无法构建环境。")

        self.bundle = bundle
        self.week_sequence = [pd.Timestamp(week) for week in week_sequence]
        self.config = config
        self.feature_frame = bundle["weekly_features"].set_index("week_start").sort_index()
        self.feature_columns = [column for column in self.feature_frame.columns if column not in {"week_start", "lt_price_source"}]
        numeric_features = self.feature_frame[self.feature_columns].astype(float)
        self.feature_mean = numeric_features.mean()
        self.feature_std = numeric_features.std(ddof=0).replace(0.0, 1.0)
        self.include_prev_reward = bool(config["env"]["include_prev_reward"])
        extra_dims = 3 if self.include_prev_reward else 2
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.feature_columns) + extra_dims,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )
        self._cursor = 0
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._prev_reward = 0.0

    def _build_observation(self, week_start: pd.Timestamp) -> np.ndarray:
        raw_features = self.feature_frame.loc[week_start, self.feature_columns].astype(float)
        normalized = ((raw_features - self.feature_mean) / self.feature_std).clip(-10.0, 10.0)
        features = normalized.to_numpy(dtype=np.float32)
        extras = [self._prev_action[0], self._prev_action[1]]
        if self.include_prev_reward:
            extras.append(self._prev_reward)
        return np.concatenate([features, np.asarray(extras, dtype=np.float32)])

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._cursor = 0
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._prev_reward = 0.0
        first_week = self.week_sequence[self._cursor]
        return self._build_observation(first_week), {}

    def step(self, action):
        week_start = self.week_sequence[self._cursor]
        summary, _, _ = simulate_week(
            bundle=self.bundle,
            week_start=week_start,
            action=(float(action[0]), float(action[1])),
            config=self.config,
            previous_lock_ratio=float(self._prev_action[0]),
        )
        reward = float(summary["reward"])
        self._prev_action = np.asarray([summary["lock_ratio"], summary["hedge_intensity"]], dtype=np.float32)
        self._prev_reward = reward
        self._cursor += 1

        terminated = self._cursor >= len(self.week_sequence)
        if terminated:
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            observation = self._build_observation(self.week_sequence[self._cursor])

        info = {
            "week_start": str(week_start),
            "procurement_cost": float(summary["procurement_cost_w"]),
            "risk_term": float(summary["risk_term_w"]),
            "trading_cost": float(summary["trans_cost_w"]),
            "hedge_error": float(summary["hedge_error_w"]),
            "cvar": float(summary["cvar_w"]),
            "reward": reward,
        }
        return observation, reward, terminated, False, info
