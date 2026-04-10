"""Environment registration."""

from gymnasium.envs.registration import register


try:
    register(id="ElecEnv-v0", entry_point="src.envs.elec_env:ElecEnv")
except Exception:
    pass
