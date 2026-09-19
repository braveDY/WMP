import gymnasium as gym

gym.register(
    id="Velocity-Rough-WMP-Train",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:UnitreeA1WMPEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agent_cfg:UnitreeA1WMPRunnerCfg",
    },
)

gym.register(
    id="Velocity-Rough-WMP-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:UnitreeA1WMPEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agent_cfg:UnitreeA1WMPRunnerCfg",
    },
)
