from functools import partial
from .multiagentenv import MultiAgentEnv, _GymWrapper
from .env_wrappers import GymWrapper
from .traffic_junction.traffic_junction_env import TrafficJunctionEnv
import gym

def env_fn(env, config) -> MultiAgentEnv:
    return env(config)


gym.register(
    id='TrafficJunction-v0',
    entry_point='envs:TrafficJunctionEnv',
)


REGISTRY = {}
REGISTRY["lbf"] = partial(env_fn, env=_GymWrapper)
REGISTRY["rware"] = partial(env_fn, env=_GymWrapper)
REGISTRY["mpe"] = partial(env_fn, env=_GymWrapper)
REGISTRY["tj"] = GymWrapper