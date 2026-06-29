# Reference: https://github.com/fan-ziqi/robot_lab

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="ATEC-Isaac-Velocity-Flat-Unitree-B2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeB2FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{ agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2FlatPPORunnerCfg"
    },
)

gym.register(
    id="ATEC-Isaac-Velocity-Rough-Unitree-B2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeB2RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{ agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2RoughPPORunnerCfg"
    },
)

gym.register(
    id="ATEC-Isaac-Velocity-Rough-Straight-Unitree-B2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeB2RoughStraightEnvCfg",
        "rsl_rl_cfg_entry_point": f"{ agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2RoughStraightPPORunnerCfg"
    },
)

gym.register(
    id="ATEC-Isaac-Velocity-Rough-Omni-B2W-Piper-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.b2w_omni_env_cfg:UnitreeB2WPiperRoughOmniEnvCfg",
        "rsl_rl_cfg_entry_point": f"{ agents.__name__}.rsl_rl_ppo_omni_cfg:UnitreeB2WPiperRoughOmniPPORunnerCfg",
    },
)

gym.register(
    id="ATEC-Isaac-Velocity-Flat-Omni-B2W-Piper-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_b2w_omni_env_cfg:UnitreeB2WPiperFlatOmniEnvCfg",
        "rsl_rl_cfg_entry_point": f"{ agents.__name__}.rsl_rl_ppo_flat_b2w_omni_cfg:UnitreeB2WPiperFlatOmniPPORunnerCfg",
    },
)

gym.register(
    id="ATEC-Isaac-TaskD-FixedArm-B2W-Easy-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.taskd_omni_env_cfg:TaskDOmniEnvEasyCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_taskd_cfg:TaskDOmniEasyPPORunnerCfg",
    },
)

gym.register(
    id="ATEC-Isaac-TaskD-FixedArm-B2W-Medium-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.taskd_omni_env_cfg:TaskDOmniEnvMediumCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_taskd_cfg:TaskDOmniMediumPPORunnerCfg",
    },
)

gym.register(
    id="ATEC-Isaac-TaskD-FixedArm-B2W-Official-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.taskd_omni_env_cfg:TaskDOmniEnvOfficialCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_taskd_cfg:TaskDOmniOfficialPPORunnerCfg",
    },
)

gym.register(
    id="ATEC-Isaac-Velocity-Flat-TaskF-Unitree-B2W-Piper-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.task_f_flat_env_cfg:UnitreeB2WTaskFFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_taskd_cfg:TaskDFlatPretrainPPORunnerCfg",
    },
)

gym.register(
    id="ATEC-Isaac-Velocity-ShortWalk-TaskF-Unitree-B2W-Piper-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.task_f_flat_env_cfg:UnitreeB2WTaskFShortWalkEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_taskd_cfg:TaskFShortWalkPPORunnerCfg",
    },
)

gym.register(
    id="ATEC-Isaac-Velocity-ShortOmni-TaskF-Unitree-B2W-Piper-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.task_f_flat_env_cfg:UnitreeB2WTaskFShortOmniEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_taskd_cfg:TaskFShortOmniPPORunnerCfg",
    },
)

gym.register(
    id="ATEC-Isaac-Velocity-ShortOmniFast-TaskF-Unitree-B2W-Piper-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.task_f_flat_env_cfg:UnitreeB2WTaskFShortOmniFastEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_taskd_cfg:TaskFShortOmniFastPPORunnerCfg",
    },
)

gym.register(
    id="ATEC-Isaac-Velocity-ShortOmniRobust-TaskF-Unitree-B2W-Piper-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.task_f_flat_env_cfg:UnitreeB2WTaskFShortOmniRobustEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_taskd_cfg:TaskFShortOmniRobustPPORunnerCfg",
    },
)
