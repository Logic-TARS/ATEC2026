from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg import UnitreeB2RoughPPORunnerCfg


@configclass
class TaskDOmniPPORunnerCfg(UnitreeB2RoughPPORunnerCfg):
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}

    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 12000
        self.experiment_name = "unitree_b2w_taskd"


@configclass
class TaskDOmniEasyPPORunnerCfg(TaskDOmniPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 2500
        self.experiment_name = "unitree_b2w_taskd_easy"


@configclass
class TaskDOmniMediumPPORunnerCfg(TaskDOmniPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 4000
        self.experiment_name = "unitree_b2w_taskd_medium"


@configclass
class TaskDOmniOfficialPPORunnerCfg(TaskDOmniPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 7000
        self.experiment_name = "unitree_b2w_taskd_official"


@configclass
class TaskDFlatPretrainPPORunnerCfg(TaskDOmniOfficialPPORunnerCfg):
    """PPO runner config for flat terrain pretraining of B2W Task D model.

    Inherits the full omni architecture (actor_hidden_dims=[512,256,128],
    critic_hidden_dims=[512,256,128], activation="elu") from
    TaskDOmniOfficialPPORunnerCfg. Overrides experiment_name and max_iterations
    for extended flat-terrain pretraining.
    """

    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 10000
        self.experiment_name = "unitree_b2w_taskd_flat_pretrain"


@configclass
class TaskFShortWalkPPORunnerCfg(TaskDOmniOfficialPPORunnerCfg):
    """PPO runner config for short-distance B2W walking pretraining."""

    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 2000
        self.experiment_name = "unitree_b2w_taskf_short_walk"


@configclass
class TaskFShortOmniPPORunnerCfg(TaskFShortWalkPPORunnerCfg):
    """Low-noise PPO runner for small-range omni fine-tuning."""

    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 300
        self.experiment_name = "unitree_b2w_taskf_short_walk"
        self.policy.init_noise_std = 0.2
        self.algorithm.entropy_coef = 0.0
        self.algorithm.learning_rate = 3.0e-4


@configclass
class TaskFShortOmniFastPPORunnerCfg(TaskFShortOmniPPORunnerCfg):
    """Lower-noise PPO runner for forward-biased short-omni acceleration."""

    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 120
        self.experiment_name = "unitree_b2w_taskf_short_walk"
        self.policy.init_noise_std = 0.15
        self.algorithm.entropy_coef = 0.0
        self.algorithm.learning_rate = 1.0e-4


@configclass
class TaskFShortOmniRobustPPORunnerCfg(TaskFShortOmniPPORunnerCfg):
    """Low-noise PPO runner for robust forward-biased short-omni fine-tuning."""

    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 80
        self.experiment_name = "unitree_b2w_taskf_short_walk"
        self.policy.init_noise_std = 0.12
        self.algorithm.entropy_coef = 0.0
        self.algorithm.learning_rate = 7.5e-5


@configclass
class TaskFShortOmniBalancedPPORunnerCfg(TaskFShortOmniRobustPPORunnerCfg):
    """Low-noise PPO runner for balanced reset-direction short-omni fine-tuning."""

    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 120
        self.experiment_name = "unitree_b2w_taskf_short_walk"
        self.policy.init_noise_std = 0.10
        self.algorithm.entropy_coef = 0.0
        self.algorithm.learning_rate = 5.0e-5


@configclass
class TaskFShortOmniDRPPORunnerCfg(TaskFShortOmniRobustPPORunnerCfg):
    """Low-noise PPO runner for light domain-randomized short-omni fine-tuning."""

    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 100
        self.experiment_name = "unitree_b2w_taskf_short_walk"
        self.policy.init_noise_std = 0.08
        self.algorithm.entropy_coef = 0.0
        self.algorithm.learning_rate = 3.0e-5
