# from .multiagentenv import MultiAgentEnv, _GymWrapper
from .tj_wrappers import TJ_Wrapper
from .traffic_junction.traffic_junction_env import TrafficJunctionEnv
from .wrappers import Wrapper
import gym
from pathlib import Path
import os
from gym.envs.registration import register, registry
from functools import partial

# def env_fn(env, config):
#     return env(config)



gym.register(
    id='TrafficJunction-v0',
    entry_point='envs.traffic_junction:TrafficJunctionEnv',
)


envs = Path(os.path.dirname(os.path.realpath(__file__))).glob("**/*_v?.py")
for e in envs:
    name = e.stem.replace("_", "-")
    lib = e.parent.stem
    filename = e.stem

    gymkey = f"pz-{lib}-{name}"
    register(
        gymkey,
        entry_point="envs.mpe:PettingZooWrapper",
        kwargs={"lib_name": lib, "env_name": filename,},
    )
    #registry.spec(gymkey).gymma_wrappers = tuple()




REGISTRY = {}
# REGISTRY["lbf"] = partial(env_fn, env=_GymWrapper)
# REGISTRY["rware"] = partial(env_fn, env=_GymWrapper)
REGISTRY["tj"] = TJ_Wrapper
REGISTRY["mpe"] = Wrapper
