"""
Vertex model for an Epithelial sheet (see definitions).

Depends on the sheet vertex geometry functions.
"""
from . import effectors
from .factory import model_factory

SheetModel = model_factory(
    [
        effectors.LineTension,
        effectors.FaceContractility,
        effectors.FaceVolumeElasticity,
    ],
    effectors.FaceVolumeElasticity,
)
