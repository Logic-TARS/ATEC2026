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
