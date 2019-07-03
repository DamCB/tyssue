import pytest
import numpy as np

from pathlib import Path

from tyssue.core.monolayer import Monolayer
from tyssue.geometry.bulk_geometry import BulkGeometry, MonolayerGeometry
from tyssue import Sheet
from tyssue.config.geometry import bulk_spec

from tyssue.generation import extrude, three_faces_sheet
from tyssue.topology.bulk_topology import IH_transition, HI_transition, remove_cell
from tyssue.topology.monolayer_topology import cell_division
from tyssue.stores import stores_dir
from tyssue.io import hdf5


def test_remove_cell():
    dsets = hdf5.load_datasets(Path(stores_dir) / "with_4sided_cell.hf5")
    mono = Monolayer("4", dsets)
    Nci = mono.Nc
    cell = mono.cell_df.query("num_faces == 4").index[0]
    res = remove_cell(mono, cell)
    MonolayerGeometry.update_all(mono)
    assert not res
    assert mono.validate()
    assert mono.Nc == Nci - 1
    with pytest.warns(UserWarning):
        cell = mono.cell_df.query("num_faces != 4").index[0]
        res = remove_cell(mono, cell)
        assert mono.validate()


def test_IH_transition():

    sheet = Sheet.planar_sheet_3d("sheet", 5, 5, 1, 1)
    sheet.sanitize()
    datasets = extrude(sheet.datasets, method="translation")

    eptm = Monolayer("test_IHt", datasets, bulk_spec())
    BulkGeometry.update_all(eptm)
    Nc, Nf, Ne, Nv = eptm.Nc, eptm.Nf, eptm.Ne, eptm.Nv
    eptm.settings["threshold_length"] = 1e-3
    IH_transition(eptm, 26)
    BulkGeometry.update_all(eptm)
    assert eptm.Nc == Nc
    assert eptm.Nf == Nf + 2
    assert eptm.Ne == Ne + 12
    assert eptm.Nv == Nv + 1

    invalid = eptm.get_invalid()
    assert np.alltrue(1 - invalid)
    assert np.alltrue(eptm.edge_df["sub_vol"] > 0)
    assert (
        eptm.face_df[eptm.face_df.segment == "apical"].shape[0] == eptm.cell_df.shape[0]
    )


def test_HI_transition():

    sheet = Sheet.planar_sheet_3d("sheet", 5, 5, 1, 1)
    sheet.sanitize()
    datasets = extrude(sheet.datasets, method="translation")

    eptm = Monolayer("test_HIt", datasets, bulk_spec())
    BulkGeometry.update_all(eptm)
    Nc, Nf, Ne, Nv = eptm.Nc, eptm.Nf, eptm.Ne, eptm.Nv
    eptm.settings["threshold_length"] = 1e-3
    IH_transition(eptm, 26)
    BulkGeometry.update_all(eptm)
    face = eptm.face_df.index[-1]
    HI_transition(eptm, face)
    assert eptm.Nc == Nc
    assert eptm.Nf == Nf
    assert eptm.Ne == Ne
    assert eptm.Nv == Nv

    invalid = eptm.get_invalid()
    assert np.alltrue(1 - invalid)
    assert np.alltrue(eptm.edge_df["sub_vol"] > 0)


def test_monolayer_division():
    datasets_2d, specs = three_faces_sheet(zaxis=True)
    datasets = extrude(datasets_2d, method="translation")
    eptm = Monolayer("test_volume", datasets, bulk_spec(), coords=["x", "y", "z"])
    MonolayerGeometry.update_all(eptm)
    for orientation in ["vertical", "horizontal"]:
        daughter = cell_division(eptm, 0, orientation=orientation)
        eptm.reset_topo()
        eptm.reset_index()

        assert eptm.validate()
    assert eptm.Nc == 5
