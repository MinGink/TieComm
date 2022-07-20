from .runner import Runner
from .runner_random import RunnerRandom
from .runner_dual import RunnerDual
from .runner_magic import RunnerMagic
from .runner_baselines import RunnerBaseline


REGISTRY = {}
REGISTRY["ac_basic"] = Runner

REGISTRY["tiecomm"] = RunnerDual

REGISTRY["tiecomm_random"] = RunnerRandom
REGISTRY["tiecomm_no"] = RunnerRandom

REGISTRY["magic"] = RunnerMagic
REGISTRY["commnet"] = RunnerBaseline
REGISTRY["ic3net"] = RunnerBaseline
REGISTRY["tarmac"] = RunnerBaseline


# REGISTRY["gacomm"] = GACommAgent