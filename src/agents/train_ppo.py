from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.backtest.simulator import simulate_strategy
from src.envs.elec_env import ElecEnv
from src.utils.plotting import save_line_plot


def resolve_torch_device(config: dict[str, Any] | None = None) -> tuple[str, dict[str, Any]]:
    requested = str((config or {}).get("device", "auto")).lower()
    mps_backend = getattr(torch.backends, "mps", None)
    capability = {
        "requested_device": requested,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "mps_built": bool(mps_backend is not None),
        "mps_available": bool(mps_backend is not None and mps_backend.is_available()),
    }

    if requested == "auto":
        if capability["cuda_available"]:
            return "cuda", capability
        if capability["mps_available"]:
            return "mps", capability
        return "cpu", capability

    if requested == "cuda":
        return ("cuda" if capability["cuda_available"] else "cpu"), capability
    if requested == "mps":
        return ("mps" if capability["mps_available"] else "cpu"), capability
    if requested == "cpu":
        return "cpu", capability
    raise ValueError(f"Unsupported torch device setting: {requested}")


class TrainingMetricsCallback(BaseCallback):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[dict[str, float]] = []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        values = dict(self.model.logger.name_to_value)
        self.records.append(
            {
                "timesteps": float(self.num_timesteps),
                "ep_rew_mean": float(values.get("rollout/ep_rew_mean", np.nan)),
                "ep_len_mean": float(values.get("rollout/ep_len_mean", np.nan)),
                "loss": float(values.get("train/loss", np.nan)),
                "value_loss": float(values.get("train/value_loss", np.nan)),
                "policy_gradient_loss": float(values.get("train/policy_gradient_loss", np.nan)),
                "entropy_loss": float(values.get("train/entropy_loss", np.nan)),
            }
        )


class SyncEvalCallback(EvalCallback):
    def _on_step(self) -> bool:
        if isinstance(self.training_env, VecNormalize) and isinstance(self.eval_env, VecNormalize):
            self.eval_env.obs_rms = self.training_env.obs_rms
            self.eval_env.ret_rms = self.training_env.ret_rms
        return super()._on_step()


def _make_env(
    bundle: dict[str, Any],
    months: list[pd.Timestamp],
    config: dict[str, Any],
    monitor_path: str | Path | None = None,
):
    def _init():
        env = ElecEnv(bundle=bundle, month_sequence=months, config=config)
        info_keywords = ("procurement_cost", "risk_term", "trading_cost", "hedge_error")
        return Monitor(env, filename=str(monitor_path) if monitor_path else None, info_keywords=info_keywords)

    return _init


def _wrap_env(vec_env, use_vec_normalize: bool, training: bool):
    if not use_vec_normalize:
        return vec_env
    return VecNormalize(vec_env, norm_obs=True, norm_reward=training, training=training)


def _export_eval_metrics(eval_log_dir: Path, output_csv: Path) -> pd.DataFrame:
    evaluation_file = eval_log_dir / "evaluations.npz"
    if not evaluation_file.exists():
        frame = pd.DataFrame(columns=["timesteps", "mean_reward", "std_reward"])
        frame.to_csv(output_csv, index=False)
        return frame

    payload = np.load(evaluation_file)
    results = payload["results"]
    timesteps = payload["timesteps"]
    frame = pd.DataFrame(
        {
            "timesteps": timesteps,
            "mean_reward": results.mean(axis=1),
            "std_reward": results.std(axis=1),
        }
    )
    frame.to_csv(output_csv, index=False)
    return frame


def _save_training_plots(train_metrics: pd.DataFrame, eval_metrics: pd.DataFrame, figure_dir: Path) -> None:
    reward_source = eval_metrics if not eval_metrics.empty else train_metrics
    reward_x = reward_source["timesteps"] if "timesteps" in reward_source else np.arange(len(reward_source))
    reward_y = reward_source["mean_reward"] if "mean_reward" in reward_source else train_metrics["ep_rew_mean"].fillna(0.0)
    save_line_plot(
        reward_x,
        reward_y,
        figure_dir / "reward_curve.png",
        title="Reward Curve",
        xlabel="Timesteps",
        ylabel="Reward",
    )

    loss_series = train_metrics["loss"].ffill().fillna(0.0) if "loss" in train_metrics else pd.Series(dtype=float)
    save_line_plot(
        train_metrics["timesteps"] if "timesteps" in train_metrics else np.arange(len(train_metrics)),
        loss_series,
        figure_dir / "loss_curve.png",
        title="Training Loss Curve",
        xlabel="Timesteps",
        ylabel="Loss",
    )


def train_model(
    bundle: dict[str, Any],
    train_months: list[pd.Timestamp],
    val_months: list[pd.Timestamp],
    config: dict[str, Any],
    output_paths: dict[str, Path],
    run_name: str = "ppo_elec_env",
    total_timesteps_override: int | None = None,
    save_plots: bool = True,
) -> dict[str, Any]:
    use_vec_normalize = bool(config.get("use_vec_normalize", False))
    tensorboard_dir = output_paths["logs"] / "tensorboard"
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    eval_log_dir = output_paths["logs"] / "eval" / run_name
    eval_log_dir.mkdir(parents=True, exist_ok=True)

    train_env = DummyVecEnv(
        [
            _make_env(
                bundle=bundle,
                months=train_months,
                config=config,
                monitor_path=output_paths["logs"] / f"{run_name}_train_monitor.csv",
            )
        ]
    )
    eval_env = DummyVecEnv(
        [
            _make_env(
                bundle=bundle,
                months=val_months,
                config=config,
                monitor_path=output_paths["logs"] / f"{run_name}_eval_monitor.csv",
            )
        ]
    )
    train_env = _wrap_env(train_env, use_vec_normalize, training=True)
    eval_env = _wrap_env(eval_env, use_vec_normalize, training=False)

    device, device_info = resolve_torch_device(config)
    model = PPO(
        policy=config["policy"],
        env=train_env,
        learning_rate=float(config["learning_rate"]),
        n_steps=int(config["n_steps"]),
        batch_size=int(config["batch_size"]),
        n_epochs=int(config["n_epochs"]),
        gamma=float(config["gamma"]),
        gae_lambda=float(config["gae_lambda"]),
        clip_range=float(config["clip_range"]),
        ent_coef=float(config["ent_coef"]),
        vf_coef=float(config["vf_coef"]),
        max_grad_norm=float(config["max_grad_norm"]),
        seed=int(config["seed"]),
        tensorboard_log=str(tensorboard_dir),
        verbose=0,
        device=device,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(int(config["checkpoint_freq"]), 1),
        save_path=str(output_paths["models"]),
        name_prefix=f"{run_name}_checkpoint",
    )
    eval_callback = SyncEvalCallback(
        eval_env=eval_env,
        best_model_save_path=str(output_paths["models"] / f"{run_name}_best"),
        log_path=str(eval_log_dir),
        eval_freq=max(int(config["eval_freq"]), 1),
        deterministic=True,
        render=False,
    )
    metrics_callback = TrainingMetricsCallback()

    model.learn(
        total_timesteps=int(total_timesteps_override or config["total_timesteps"]),
        callback=CallbackList([checkpoint_callback, eval_callback, metrics_callback]),
        progress_bar=False,
    )

    model_path = output_paths["models"] / f"{run_name}.zip"
    model.save(model_path)
    if use_vec_normalize:
        train_env.save(str(output_paths["models"] / "vecnormalize.pkl"))

    train_metrics = pd.DataFrame(metrics_callback.records)
    train_metrics_name = "train_metrics.csv" if run_name == "ppo_elec_env" else f"{run_name}_train_metrics.csv"
    eval_metrics_name = "eval_metrics.csv" if run_name == "ppo_elec_env" else f"{run_name}_eval_metrics.csv"
    train_metrics.to_csv(output_paths["metrics"] / train_metrics_name, index=False)
    eval_metrics = _export_eval_metrics(eval_log_dir, output_paths["metrics"] / eval_metrics_name)
    if save_plots:
        _save_training_plots(train_metrics, eval_metrics, output_paths["figures"])

    return {
        "model": model,
        "model_path": model_path,
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
        "best_model_path": output_paths["models"] / f"{run_name}_best" / "best_model.zip",
        "device": device,
        "device_info": device_info,
    }


def load_model(model_path: str | Path, config: dict[str, Any] | None = None) -> PPO:
    device, _ = resolve_torch_device(config)
    return PPO.load(str(model_path), device=device)


def collect_policy_actions(
    model: PPO,
    bundle: dict[str, Any],
    months: list[pd.Timestamp],
    config: dict[str, Any],
) -> dict[pd.Timestamp, tuple[float, float]]:
    env = ElecEnv(bundle=bundle, month_sequence=months, config=config)
    observation, _ = env.reset()
    actions: dict[pd.Timestamp, tuple[float, float]] = {}
    done = False
    while not done:
        current_month = env.month_sequence[env._cursor]
        action, _ = model.predict(observation, deterministic=True)
        actions[current_month] = (float(action[0]), float(action[1]))
        observation, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    return actions


def evaluate_policy(
    model: PPO,
    bundle: dict[str, Any],
    months: list[pd.Timestamp],
    config: dict[str, Any],
    strategy_name: str = "ppo_policy",
) -> dict[str, Any]:
    actions = collect_policy_actions(model, bundle, months, config)
    result = simulate_strategy(
        bundle=bundle,
        months=months,
        action_source=actions,
        config=config,
        strategy_name=strategy_name,
    )
    result["actions"] = actions
    return result
