from .runner import Runner
from .runner_gnn import RunnerGNN



from .runner_magic import RunnerMagic
from .runner_baselines import RunnerBaseline
from collections import namedtuple
from .runner_ic3net import RunnerIcnet

from .runner_tiecomm import RunnerTiecomm
from .runner_default import RunnerDefualt







REGISTRY = {}
REGISTRY["ac_mlp"] = Runner
REGISTRY["ac_att"] = Runner
REGISTRY["gnn"] = RunnerGNN

REGISTRY["tiecomm"] = RunnerTiecomm
REGISTRY["tiecomm_default"] = RunnerDefualt



REGISTRY["magic"] = RunnerMagic
REGISTRY["commnet"] = RunnerBaseline
REGISTRY["ic3net"] = RunnerIcnet
REGISTRY["tarmac"] = RunnerBaseline
# REGISTRY["gacomm"] = GACommAgent