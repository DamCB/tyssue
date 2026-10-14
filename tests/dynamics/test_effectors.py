import numpy as np

import pytest

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


def test_active_migration_direction():
    from tyssue.dynamics import model_factory
    from tyssue.dynamics.effectors import ActiveMigration

    model = model_factory([ActiveMigration])
    sheet = Sheet.planar_sheet_2d("planar", 3, 3, 1, 1)
    sheet.sanitize()
    sheet.update_specs(model.specs, reset=True)
    sheet.reset_topo()  # num_sides was reset by update_specs
    PlanarGeometry.update_all(sheet)
    # the default direction comes from the specs
    assert (sheet.face_df["mx"] == 1.0).all()

    sheet.face_df["mx"] = 0.0
    sheet.face_df["my"] = 2.0
    grad = model.compute_gradient(sheet)
    assert np.allclose(grad["gx"], 0.0)
    assert (grad["gy"] <= 0).all()
    # each cell's total force is migration_strength, whatever its sides
    strength = sheet.face_df["migration_strength"]
    assert np.isclose(-grad["gy"].sum(), strength.sum())
