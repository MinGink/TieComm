from .tiecomm import TieCommAgent
from .commnet import CommNetAgent
from .tarmac import TarCommAgent
from .magic import MAGICAgent
# from .gacomm import GACommAgent
from .models import MLP


REGISTRY = {}
REGISTRY["tiecomm"] = TieCommAgent
REGISTRY["commnet"] = CommNetAgent
REGISTRY["ic3net"] = CommNetAgent
# REGISTRY["gacomm"] = GACommAgent
REGISTRY["tarmac"] = TarCommAgent
REGISTRY["magic"] = MAGICAgent
REGISTRY["ac_basic"] = MLP