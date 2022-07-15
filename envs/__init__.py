from functools import partial
from .multiagentenv import MultiAgentEnv, _GymWrapper
from .env_wrappers import GymWrapper
from .traffic_junction.traffic_junction_env import TrafficJunctionEnv


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {}
REGISTRY["lbf"] = partial(env_fn, env=_GymWrapper)
REGISTRY["rware"] = partial(env_fn, env=_GymWrapper)
REGISTRY["mpe"] = partial(env_fn, env=_GymWrapper)
