"""The `generation` module provides utilities to easily create :class:`Epithelium` objects.
"""

from .utils import *
from .hexagonal_grids import *
from .from_voronoi import *
from .modifiers import *
from .shapes import *

try:
    from .cpp import mesh_generation
except ImportError:
    print("C++ extension are not available for this version")
