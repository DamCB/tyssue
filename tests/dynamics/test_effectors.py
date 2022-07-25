from tyssue import Monolayer, MonolayerGeometry, Sheet, config
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
