from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg import UnitreeB2RoughPPORunnerCfg


@configclass
class UnitreeB2WPiperRoughOmniPPORunnerCfg(UnitreeB2RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 20000
        self.experiment_name = "unitree_b2w_rough_omni"
        # Actor/critic architecture, PPO hyperparams, save interval,
        # and everything else inherited from UnitreeB2RoughPPORunnerCfg.
