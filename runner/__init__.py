from .runner import Runner
from .runner_random import RunnerRandom
from .runner_one import RunnerOne
from .runner_dual import RunnerDual
from .runner_magic import RunnerMagic
from .runner_baselines import RunnerBaseline
from collections import namedtuple
from .runner_ic3net import RunnerIcnet








REGISTRY = {}
REGISTRY["ac_mlp"] = Runner
REGISTRY["ac_att"] = Runner

REGISTRY["tiecomm"] = RunnerDual
REGISTRY["tiecomm_random"] = RunnerRandom
REGISTRY["tiecomm_one"] = RunnerOne

REGISTRY["magic"] = RunnerMagic
REGISTRY["commnet"] = RunnerBaseline
REGISTRY["ic3net"] = RunnerIcnet
REGISTRY["tarmac"] = RunnerBaseline


# REGISTRY["gacomm"] = GACommAgent