from . import effectors
from .factory import model_factory

PlanarModel = model_factory(
    [effectors.LineTension, effectors.FaceContractility, effectors.FaceAreaElasticity],
    effectors.FaceAreaElasticity,
)
