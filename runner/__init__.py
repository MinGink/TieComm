from .runner import Runner
from .runner_random import RunnerRandom
from .runner_dual import RunnerDual
from .runner_magic import RunnerMagic


REGISTRY = {}
REGISTRY["tiecomm"] = RunnerDual
REGISTRY["magic"] = RunnerMagic
REGISTRY["tiecomm_random"] = RunnerRandom
REGISTRY["tiecomm_no"] = RunnerRandom

REGISTRY["commnet"] = Runner
REGISTRY["ic3net"] = Runner
REGISTRY["tarmac"] = Runner
REGISTRY["ac_basic"] = Runner

# REGISTRY["gacomm"] = GACommAgent