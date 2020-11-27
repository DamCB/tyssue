from pathlib import Path

from tyssue.generation import extrude, three_faces_sheet
from tyssue import Monolayer, config, Sheet, MonolayerGeometry
from tyssue.stores import stores_dir
from tyssue.io.hdf5 import load_datasets
from tyssue.utils import testing
from tyssue.dynamics.bulk_model import (
    BulkModel,
    BulkModelwithFreeBorders,
    ClosedMonolayerModel,
)

from tyssue.dynamics.effectors import BorderElasticity


def test_effector():

    sheet_dsets, specs = three_faces_sheet()
    sheet = Sheet("test", sheet_dsets, specs)
    mono = Monolayer.from_flat_sheet("test", sheet, config.geometry.bulk_spec())
    MonolayerGeometry.update_all(mono)
    testing.effector_tester(mono, BorderElasticity)


def test_models():

    sheet_dsets, specs = three_faces_sheet()
    sheet = Sheet("test", sheet_dsets, specs)
    mono = Monolayer.from_flat_sheet("test", sheet, config.geometry.bulk_spec())
    MonolayerGeometry.update_all(mono)

    testing.model_tester(mono, BulkModel)
    testing.model_tester(mono, BulkModelwithFreeBorders)
    datasets = load_datasets(
        Path(stores_dir) / "small_ellipsoid.hf5",
        data_names=["vert", "edge", "face", "cell"],
    )

    specs = config.geometry.bulk_spec()
    monolayer = Monolayer("ell", datasets, specs)
    MonolayerGeometry.update_all(mono)

    testing.model_tester(monolayer, ClosedMonolayerModel)
