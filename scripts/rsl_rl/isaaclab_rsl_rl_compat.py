from importlib import import_module
from typing import Any

_rsl_rl_module = import_module("isaaclab_rl.rsl_rl")

RslRlBaseRunnerCfg = _rsl_rl_module.RslRlBaseRunnerCfg
RslRlVecEnvWrapper = _rsl_rl_module.RslRlVecEnvWrapper


def _identity(value: Any, installed_version: str) -> Any:
    return value


handle_deprecated_rsl_rl_cfg = getattr(
    _rsl_rl_module, "handle_deprecated_rsl_rl_cfg", _identity
)
handle_deprecated_rsl_rl_checkpoint = getattr(
    _rsl_rl_module, "handle_deprecated_rsl_rl_checkpoint", _identity
)

if not hasattr(RslRlVecEnvWrapper, "get_privileged_observations"):
    def _get_privileged_observations(self):
        obs = self.get_observations()
        if hasattr(obs, "get"):
            return obs.get("critic", None)
        return getattr(self, "privileged_obs_buf", None)
    RslRlVecEnvWrapper.get_privileged_observations = _get_privileged_observations
