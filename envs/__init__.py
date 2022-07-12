from functools import partial
from .multiagentenv import MultiAgentEnv, _GymWrapper

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)





REGISTRY = {}
REGISTRY["lbf"] = partial(env_fn, env=_GymWrapper)
REGISTRY["rware"] = partial(env_fn, env=_GymWrapper)
REGISTRY["mpe"] = partial(env_fn, env=_GymWrapper)
# REGISTRY["tj"] = partial(env_fn, env=_GymWrapper)
