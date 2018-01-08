from . import effectors
from .factory import model_factory

PlanarModel = model_factory(
    [effectors.LineTension,
     effectors.FaceContractility,
     effectors.FaceAreaElasticity],
    effectors.FaceAreaElasticity)

PlanarModel.__doc__ = """ Model for a 2D junction network  in 2D.

""" + PlanarModel.__doc__
