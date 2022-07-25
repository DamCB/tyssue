import numpy as np
from pytest import warns

from tyssue import Sheet
from tyssue.dynamics import effectors, factory, units
from tyssue.generation import three_faces_sheet


def test_old_style_warn():
    class BarrierElasticity(effectors.AbstractEffector):
        """
        Barrier use to maintain the tissue integrity.
        """

        dimensions = units.line_elasticity
        magnitude = "barrier_elasticity"
        label = "Barrier elasticity"
        element = "vert"
        specs = {
            "vert": {"barrier_elasticity", "is_active", "delta_rho"}
        }  # distance to a barrier membrane

        @staticmethod
        def energy(eptm):
            return eptm.vert_df.eval("delta_rho**2 * barrier_elasticity/2")

        @staticmethod
        def gradient(eptm):
            grad = np.zeros((eptm.Nv, 3))
            grad.columns = ["g" + c for c in eptm.coords]
            return grad, None

    with warns(UserWarning):
        factory.model_factory([BarrierElasticity, effectors.FaceAreaElasticity])


def test_update_specs():
    class BarrierElasticity(effectors.AbstractEffector):
        """
        Barrier use to maintain the tissue integrity.
        """

        dimensions = units.line_elasticity
        magnitude = "barrier_elasticity"
        label = "Barrier elasticity"
        element = "vert"
        specs = {
            "vert": {"barrier_elasticity": 1.2, "is_active": 1, "delta_rho": 0.0}
        }  # distance to a barrier membrane

        @staticmethod
        def energy(eptm):
            return eptm.vert_df.eval("delta_rho**2 * barrier_elasticity/2")

        @staticmethod
        def gradient(eptm):
            grad = np.zeros((eptm.Nv, 3))
            grad.columns = ["g" + c for c in eptm.coords]
            return grad, None

    model = factory.model_factory(
        [
            BarrierElasticity,
            effectors.LineTension,
            effectors.FaceAreaElasticity,
            effectors.FaceContractility,
        ]
    )
    sheet_dsets, specs = three_faces_sheet()
    sheet = Sheet("test", sheet_dsets, specs)
    sheet.update_specs(model.specs)
    assert sheet.vert_df.loc[0, "barrier_elasticity"] == 1.2
