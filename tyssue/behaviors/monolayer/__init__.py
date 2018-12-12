"""Monolayer related behaviors

For now only apoptosis is defined, as a sequence of actions leading to the cell
disapearing from a monolayer
"""

from .apoptosis_events import apoptosis
from .actions import (grow,
                      shrink,
                      contract,
                      contract_apical_face,
                      ab_pull)
