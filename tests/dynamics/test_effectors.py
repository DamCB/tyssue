import numpy as np
from tyssue import Monolayer, MonolayerGeometry, Sheet, config, PlanarGeometry
from tyssue.dynamics.effectors import (
    BarrierElasticity,
    BorderElasticity,
    CellAreaElasticity,
    CellVolumeElasticity,
    FaceAreaElasticity,
    FaceContractility,
    FaceVolumeElasticity,
    LengthElasticity,
    LineTension,
    PerimeterElasticity,
    RadialTension,
    SurfaceTension,
    #    Repulsion
)
from tyssue.generation import three_faces_sheet
from tyssue.utils import testing

sheet_effectors = [
    LengthElasticity,
    PerimeterElasticity,
    FaceAreaElasticity,
    FaceVolumeElasticity,
    LineTension,
    FaceContractility,
    SurfaceTension,
    BorderElasticity,
    RadialTension,
    BarrierElasticity,
]

bulk_effectors = [CellAreaElasticity, CellVolumeElasticity]


def test_effectors():
    sheet_dsets, specs = three_faces_sheet()
    sheet = Sheet("test", sheet_dsets, specs)
    mono = Monolayer.from_flat_sheet("test", sheet, config.geometry.bulk_spec())
    MonolayerGeometry.update_all(mono)

    for effector in sheet_effectors:
        testing.effector_tester(sheet, effector)

    for effector in bulk_effectors:
        testing.effector_tester(mono, effector)


# def test_repulsion():
#     sheet_dsets, specs = three_faces_sheet()
#     specs['vert']['force_repulsion'] = 10
#     sheet = Sheet("test", sheet_dsets, specs, coords=list('xy'))
#     PlanarGeometry.update_all(sheet)
#     PlanarGeometry.update_repulsion(sheet)

#     energy = Repulsion.energy(sheet)
#     assert energy.shape == (sheet.datasets[Repulsion.element].shape[0],)
#     assert np.all(np.isfinite(energy))

#     grad_s, grad_t = Repulsion.gradient(sheet)
#     assert np.all(np.isfinite(grad_s))

#     if grad_s.shape == (sheet.Nv, sheet.dim):
#         assert grad_t is None
#     elif grad_s.shape == (sheet.Ne, sheet.dim):
#         assert grad_t.shape == (sheet.Ne, sheet.dim)
#         assert np.all(np.isfinite(grad_t))
#     else:
#         raise ValueError(
#             f"""
#             The computed gradients for effector {Repulsion.label}
#             should have shape {(sheet.Ne, sheet.dim)} or {(sheet.Nv, sheet.dim)},
#             found {grad_s.shape}.
#             """
#         )
