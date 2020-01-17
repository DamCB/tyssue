"""
Specific functions for apoptosis vertex model
"""

from ..utils import to_nd
from . import units

from .sheet_gradients import height_grad
from . import effectors
from .factory import model_factory


class ApicoBasalTension(effectors.AbstractEffector):
    """ Effector for the apical-basal tension.

    The energy is proportional to the heigth of the cell
    """

    dimensions = units.line_elasticity
    label = "Apical-basal tension"
    magnitude = "radial_tension"
    element = "vert"
    specs = {"vert": {"radial_tension":0.0, "height": 1.0, "is_active": 1}}

    @staticmethod
    def energy(sheet):
        return sheet.vert_df.eval("height * radial_tension * is_active")

    @staticmethod
    def gradient(sheet):
        grad = to_nd(sheet.vert_df.eval("radial_tension * is_active"), 3) * height_grad(
            sheet
        )
        grad.columns = ["gx", "gy", "gz"]
        return grad, None


SheetApoptosisModel = model_factory(
    [
        effectors.LineTension,
        effectors.FaceContractility,
        ApicoBasalTension,
        effectors.FaceVolumeElasticity,
    ],
    effectors.FaceVolumeElasticity,
)
