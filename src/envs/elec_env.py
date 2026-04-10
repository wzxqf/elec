from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from src.backtest.simulator import simulate_month


class ElecEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        bundle: dict[str, Any],
        month_sequence: list[pd.Timestamp],
        config: dict[str, Any],
    ) -> None:
        super().__init__()
        if not month_sequence:
            raise ValueError("month_sequence 为空，无法构建环境。")
        self.bundle = bundle
        self.month_sequence = [pd.Timestamp(month) for month in month_sequence]
        self.config = config
        self.feature_frame = bundle["monthly_features"].set_index("month").sort_index()
        self.feature_columns = list(self.feature_frame.columns)
        self.obs_columns = self.feature_columns + [
            "prev_lock_ratio",
            "prev_hedge_intensity",
            "prev_reward",
        ]
        obs_dim = len(self.obs_columns)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-1e9,
            high=1e9,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self._cursor = 0
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._prev_reward = 0.0
        self._episode_records: list[dict[str, Any]] = []

    def _build_observation(self, month: pd.Timestamp) -> np.ndarray:
        features = self.feature_frame.loc[month, self.feature_columns].to_numpy(dtype=np.float32)
        tail = np.array(
            [self._prev_action[0], self._prev_action[1], self._prev_reward],
            dtype=np.float32,
        )
        return np.concatenate([features, tail]).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._cursor = 0
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._prev_reward = 0.0
        self._episode_records = []
        first_month = self.month_sequence[self._cursor]
        return self._build_observation(first_month), {}

    def step(self, action: np.ndarray):
        month = self.month_sequence[self._cursor]
        summary, _ = simulate_month(self.bundle, month, tuple(action.tolist()), self.config)
        reward = float(summary["reward"])
        self._episode_records.append(summary)
        self._prev_action = np.asarray(action, dtype=np.float32)
        self._prev_reward = reward
        self._cursor += 1
        terminated = self._cursor >= len(self.month_sequence)
        if terminated:
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            observation = self._build_observation(self.month_sequence[self._cursor])

        info = {
            "month": str(month),
            "procurement_cost": summary["procurement_cost_m"],
            "risk_term": summary["risk_term_m"],
            "trading_cost": summary["trading_cost_m"],
            "hedge_error": summary["hedge_error_m"],
            "lock_ratio": summary["lock_ratio"],
            "hedge_intensity": summary["hedge_intensity"],
        }
        return observation, reward, terminated, False, info
