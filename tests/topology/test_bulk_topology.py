from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import Voronoi

from tyssue import Epithelium, Monolayer, Sheet
from tyssue.config.geometry import bulk_spec
from tyssue.generation import extrude, from_3d_voronoi, hexa_grid3d, three_faces_sheet
from tyssue.geometry.bulk_geometry import BulkGeometry, MonolayerGeometry
from tyssue.io import hdf5
from tyssue.stores import stores_dir
from tyssue.topology.bulk_topology import (
    HI_transition,
    IH_transition,
    cell_division,
    close_cell,
    find_HIs,
    find_IHs,
    find_rearangements,
    fix_pinch,
    remove_cell,
    split_vert,
)
from tyssue.topology.monolayer_topology import cell_division as monolayer_division


def test_bulk_division():

    cells = hexa_grid3d(4, 4, 6)
    datasets = from_3d_voronoi(Voronoi(cells))
    specs = bulk_spec()
    bulk = Epithelium("bulk", datasets, specs)
    bulk.reset_topo()
    bulk.reset_index()
    bulk.sanitize()
    bulk.reset_topo()
    bulk.reset_index()
    cell_division(bulk, 4, BulkGeometry)

    dsets = hdf5.load_datasets(Path(stores_dir) / "with_4sided_cell.hf5")
    bulk = Monolayer("4", dsets)

    BulkGeometry.update_all(bulk)

    # daughter = cell_division(bulk, 12, BulkGeometry)
    with pytest.warns(UserWarning):
        cell_division(bulk, 4, BulkGeometry)
        assert bulk.validate()


def test_close_cell():

    sheet = Sheet.planar_sheet_3d("sheet", 5, 5, 1, 1)
    sheet.sanitize()
    datasets = extrude(sheet.datasets, method="translation")

    eptm = Monolayer("test_IHt", datasets, bulk_spec())
    BulkGeometry.update_all(eptm)
    Nf = eptm.Nf
    cell = 5
    cell_nf = eptm.cell_df.loc[cell, "num_faces"]

    face = eptm.edge_df.loc[eptm.edge_df.cell == cell, "face"].iloc[0]

    eptm.face_df.drop(face, axis=0, inplace=True)
    edges = eptm.edge_df[eptm.edge_df.face == face].index

    eptm.edge_df.drop(edges, axis=0, inplace=True)
    eptm.reset_topo()
    eptm.reset_index()

    assert Nf == eptm.Nf + 1
    assert cell_nf == eptm.cell_df.loc[cell, "num_faces"] + 1

    close_cell(eptm, cell)
    assert Nf == eptm.Nf
    assert cell_nf == eptm.cell_df.loc[cell, "num_faces"]
    assert np.all(np.isfinite(eptm.face_df.x))


def test_close_already_closed():

    dsets = hdf5.load_datasets(Path(stores_dir) / "with_4sided_cell.hf5")
    mono = Monolayer("4", dsets)
    cell = mono.cell_df.query("num_faces != 4").index[0]
    close_cell(mono, cell)


def test_close_two_holes():
    dsets = hdf5.load_datasets(Path(stores_dir) / "small_ellipsoid.hf5")
    mono = Monolayer("4", dsets)
    cell = mono.cell_df.query("num_faces != 4").index[0]
    edges = mono.edge_df.query(f"cell == {cell}")
    faces = edges["face"].iloc[[0, 8]]
    face_edges = edges[edges["face"].isin(faces)].index
    mono.face_df.drop(faces, axis=0, inplace=True)
    mono.edge_df.drop(face_edges, axis=0, inplace=True)
    mono.reset_index()
    mono.reset_topo()
    with pytest.raises(ValueError):
        close_cell(mono, cell)


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
    assert np.all(1 - invalid)
    assert np.all(eptm.edge_df["sub_vol"] > 0)
    assert (
        eptm.face_df[eptm.face_df.segment == "apical"].shape[0] == eptm.cell_df.shape[0]
    )


def test_split_vert():
    sheet = Sheet.planar_sheet_3d("sheet", 5, 5, 1, 1)
    sheet.sanitize()
    datasets = extrude(sheet.datasets, method="translation")

    eptm = Monolayer("test_IHt", datasets, bulk_spec())
    BulkGeometry.update_all(eptm)

    split_vert(eptm, 32, face=None, multiplier=1.5)

    BulkGeometry.update_all(eptm)

    invalid = eptm.get_invalid()
    assert np.all(1 - invalid)


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
    assert np.all(1 - invalid)
    assert np.all(eptm.edge_df["sub_vol"] > 0)


def test_find_transitions():

    sheet = Sheet.planar_sheet_3d("sheet", 5, 5, 1, 1)
    sheet.sanitize()
    datasets = extrude(sheet.datasets, method="translation")

    eptm = Monolayer("test_IHt", datasets, bulk_spec())
    BulkGeometry.update_all(eptm)
    eptm.settings["threshold_length"] = 1e-3
    IH_transition(eptm, 26)
    BulkGeometry.update_all(eptm)

    eptm.settings["threshold_length"] = 1e-2
    ih, hi = find_rearangements(eptm)

    assert len(ih) == 0
    assert len(hi) == 2
    assert len(find_HIs(eptm))
    assert len(find_IHs(eptm)) == 0

    face = eptm.face_df.index[-1]
    HI_transition(eptm, face)
    BulkGeometry.update_all(eptm)

    eptm.settings["threshold_length"] = 2e-1

    ih, hi = find_rearangements(eptm)
    assert len(ih) == 1
    assert len(hi) == 0


def test_monolayer_division():
    datasets_2d, _ = three_faces_sheet(zaxis=True)
    datasets = extrude(datasets_2d, method="translation")
    eptm = Monolayer("test_volume", datasets, bulk_spec(), coords=["x", "y", "z"])
    eptm.vert_df[eptm.coords] += np.random.normal(scale=1e-6, size=(eptm.Nv, 3))
    MonolayerGeometry.update_all(eptm)
    for orientation in ["vertical", "horizontal", "apical"]:
        monolayer_division(eptm, 0, orientation=orientation)
        eptm.reset_topo()
        eptm.reset_index()

        assert eptm.validate()
    assert eptm.Nc == 6


def test_fix_pinch():
    dsets = hdf5.load_datasets(Path(stores_dir) / "with_pinch.hf5")
    pinched = Monolayer("pinched", dsets)
    assert not pinched.validate()
    fix_pinch(pinched)
    assert pinched.validate()
    edf = pinched.edge_df[["srce", "trgt", "face", "cell"]].copy()
    # Nothing happens here
    fix_pinch(pinched)
    assert pinched.validate()
    assert np.all(pinched.edge_df[["srce", "trgt", "face", "cell"]] == edf)
