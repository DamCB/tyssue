"""
Dynamical models for monlayer and bulk epithelium.


"""

from . import effectors
from .factory import model_factory


BulkModel = model_factory(
    [
        effectors.LineTension,
        effectors.FaceContractility,
        effectors.CellAreaElasticity,
        effectors.CellVolumeElasticity,
    ],
    effectors.CellVolumeElasticity,
)

ClosedMonolayerModel = model_factory(
    [
        effectors.LineTension,
        effectors.FaceContractility,
        effectors.CellAreaElasticity,
        effectors.CellVolumeElasticity,
        effectors.LumenVolumeElasticity,
    ],
    effectors.CellVolumeElasticity,
)


BulkModelwithFreeBorders = model_factory(
    [
        effectors.LineTension,
        effectors.FaceContractility,
        effectors.CellAreaElasticity,
        effectors.BorderElasticity,
        effectors.CellVolumeElasticity,
    ],
    effectors.CellVolumeElasticity,
)


class LaminaModel(BulkModel):
    """Not implemented yet
    """

    pass
