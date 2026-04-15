from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.backtest.simulator import simulate_strategy
from src.envs.elec_env import ElecEnv


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
    weeks: list[pd.Timestamp],
    config: dict[str, Any],
    monitor_path: str | Path | None = None,
):
    def _init():
        env = ElecEnv(bundle=bundle, week_sequence=weeks, config=config)
        info_keywords = ("procurement_cost", "risk_term", "trading_cost", "hedge_error", "cvar")
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
    frame = pd.DataFrame(
        {
            "timesteps": payload["timesteps"],
            "mean_reward": payload["results"].mean(axis=1),
            "std_reward": payload["results"].std(axis=1),
        }
    )
    frame.to_csv(output_csv, index=False)
    return frame


def _save_training_plots(train_metrics: pd.DataFrame, eval_metrics: pd.DataFrame, figure_dir: Path) -> None:
    from src.utils.plotting import save_line_plot

    reward_source = eval_metrics if not eval_metrics.empty else train_metrics
    reward_x = reward_source["timesteps"] if "timesteps" in reward_source else np.arange(len(reward_source))
    reward_y = reward_source["mean_reward"] if "mean_reward" in reward_source else train_metrics["ep_rew_mean"].fillna(0.0)
    save_line_plot(
        reward_x,
        reward_y,
        figure_dir / "weekly_reward_curve.png",
        title="周度奖励曲线",
        xlabel="训练步数",
        ylabel="平均奖励",
    )
    save_line_plot(
        train_metrics["timesteps"] if "timesteps" in train_metrics else np.arange(len(train_metrics)),
        train_metrics["loss"].ffill().fillna(0.0) if "loss" in train_metrics else [],
        figure_dir / "loss_curve.png",
        title="训练损失曲线",
        xlabel="训练步数",
        ylabel="损失值",
    )


def train_model(
    bundle: dict[str, Any],
    train_weeks: list[pd.Timestamp],
    val_weeks: list[pd.Timestamp],
    config: dict[str, Any],
    output_paths: dict[str, Path],
    run_name: str = "ppo",
    total_timesteps_override: int | None = None,
    save_plots: bool = True,
    persist_artifacts: bool = True,
) -> dict[str, Any]:
    use_vec_normalize = bool(config.get("use_vec_normalize", False))
    tensorboard_dir = output_paths["logs"] / "tensorboard" if persist_artifacts else None
    eval_log_dir = output_paths["logs"] / "eval" / run_name if persist_artifacts else None
    if tensorboard_dir is not None:
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
    if eval_log_dir is not None:
        eval_log_dir.mkdir(parents=True, exist_ok=True)

    train_env = DummyVecEnv(
        [
            _make_env(
                bundle=bundle,
                weeks=train_weeks,
                config=config,
                monitor_path=output_paths["logs"] / f"{run_name}_train_monitor.csv" if persist_artifacts else None,
            )
        ]
    )
    eval_env = None
    if persist_artifacts:
        eval_env = DummyVecEnv(
            [
                _make_env(
                    bundle=bundle,
                    weeks=val_weeks,
                    config=config,
                    monitor_path=output_paths["logs"] / f"{run_name}_eval_monitor.csv",
                )
            ]
        )
    train_env = _wrap_env(train_env, use_vec_normalize, training=True)
    if eval_env is not None:
        eval_env = _wrap_env(eval_env, use_vec_normalize, training=False)

    device = str(config.get("device", "cpu"))
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
        tensorboard_log=str(tensorboard_dir) if tensorboard_dir is not None else None,
        verbose=0,
        device=device,
    )

    metrics_callback = TrainingMetricsCallback()
    callbacks: list[BaseCallback] = [metrics_callback]
    if persist_artifacts:
        checkpoint_callback = CheckpointCallback(
            save_freq=max(int(config["checkpoint_freq"]), 1),
            save_path=str(output_paths["models"]),
            name_prefix=f"{run_name}_checkpoint",
        )
        eval_callback = SyncEvalCallback(
            eval_env=eval_env,
            best_model_save_path=str(output_paths["models"] / f"{run_name}_best_tmp"),
            log_path=str(eval_log_dir),
            eval_freq=max(int(config["eval_freq"]), 1),
            deterministic=True,
            render=False,
            verbose=0,
        )
        callbacks = [checkpoint_callback, eval_callback, metrics_callback]

    model.learn(
        total_timesteps=int(total_timesteps_override or config["total_timesteps"]),
        callback=CallbackList(callbacks),
        progress_bar=False,
    )

    latest_model_path = output_paths["models"] / f"{run_name}_latest.zip"
    best_model_path = output_paths["models"] / f"{run_name}_best.zip"
    if persist_artifacts:
        model.save(latest_model_path)
        tmp_best = output_paths["models"] / f"{run_name}_best_tmp" / "best_model.zip"
        if tmp_best.exists():
            shutil.copyfile(tmp_best, best_model_path)
        else:
            model.save(best_model_path)
        if use_vec_normalize:
            train_env.save(str(output_paths["models"] / f"{run_name}_vecnormalize.pkl"))

    train_metrics = pd.DataFrame(metrics_callback.records)
    if persist_artifacts:
        train_metrics.to_csv(output_paths["metrics"] / f"{run_name}_train_metrics.csv", index=False)
        eval_metrics = _export_eval_metrics(eval_log_dir, output_paths["metrics"] / f"{run_name}_eval_metrics.csv")
    else:
        eval_metrics = pd.DataFrame(columns=["timesteps", "mean_reward", "std_reward"])
    if save_plots and persist_artifacts:
        _save_training_plots(train_metrics, eval_metrics, output_paths["figures"])

    return {
        "model": model,
        "model_path": latest_model_path if persist_artifacts else None,
        "best_model_path": best_model_path if persist_artifacts else None,
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
        "device": device,
        "gpu_used": device.startswith("cuda"),
        "run_name": run_name,
    }


def load_model(model_path: str | Path, device: str = "cpu") -> PPO:
    return PPO.load(str(model_path), device=device)


def collect_policy_actions(
    model: PPO,
    bundle: dict[str, Any],
    weeks: list[pd.Timestamp],
    config: dict[str, Any],
) -> dict[pd.Timestamp, tuple[float, float]]:
    env = ElecEnv(bundle=bundle, week_sequence=weeks, config=config)
    observation, _ = env.reset()
    actions: dict[pd.Timestamp, tuple[float, float]] = {}
    done = False
    while not done:
        current_week = env.week_sequence[env._cursor]
        action, _ = model.predict(observation, deterministic=True)
        actions[current_week] = (float(action[0]), float(action[1]))
        observation, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    return actions


def evaluate_policy(
    model: PPO,
    bundle: dict[str, Any],
    weeks: list[pd.Timestamp],
    config: dict[str, Any],
    strategy_name: str = "ppo_policy",
) -> dict[str, Any]:
    actions = collect_policy_actions(model, bundle, weeks, config)
    result = simulate_strategy(
        bundle=bundle,
        weeks=weeks,
        action_source=actions,
        config=config,
        strategy_name=strategy_name,
    )
    result["actions"] = actions
    return result
