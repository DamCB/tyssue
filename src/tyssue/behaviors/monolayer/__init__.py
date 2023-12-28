"""Monolayer related behaviors

For now only apoptosis is defined, as a sequence of actions leading to the cell
disapearing from a monolayer
"""

from .actions import ab_pull, contract, contract_apical_face, grow, shrink  # noqa
from .apoptosis_events import apoptosis  # noqa
from .basic_events import contraction  # noqa
from .delamination_events import constriction  # noqa
