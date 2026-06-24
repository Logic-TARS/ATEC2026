from isaaclab.utils import configclass

from .rsl_rl_ppo_omni_cfg import UnitreeB2WPiperRoughOmniPPORunnerCfg


@configclass
class UnitreeB2WPiperFlatOmniPPORunnerCfg(UnitreeB2WPiperRoughOmniPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 10000
        self.experiment_name = "unitree_b2w_flat_omni"
        # Actor/critic architecture [512, 256, 128] + ELU activation and PPO hyperparams
        # are inherited from UnitreeB2WPiperRoughOmniPPORunnerCfg → UnitreeB2RoughPPORunnerCfg.
