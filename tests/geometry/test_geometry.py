import os

import numpy as np
import pytest

from tyssue import config
from tyssue.core.sheet import Sheet
from tyssue.generation import three_faces_sheet
from tyssue.geometry.sheet_geometry import SheetGeometry as sgeom
from tyssue.io.hdf5 import load_datasets
from tyssue.stores import stores_dir
from tyssue.topology.base_topology import collapse_edge


def test_3faces():

    datasets, specs = three_faces_sheet()
    sheet = Sheet("3faces_2D", datasets, specs)
    sgeom.update_dcoords(sheet)
    sgeom.update_length(sheet)
    np.testing.assert_allclose(sheet.edge_df["length"], 1, rtol=1e-3)

    sgeom.update_centroid(sheet)
    np.testing.assert_array_almost_equal(
        [0.5, -1.0, 0.5], sheet.face_df["x"], decimal=3
    )

    sgeom.update_normals(sheet)
    norms = np.linalg.norm(sheet.edge_df[["nx", "ny", "nz"]], axis=1)
    np.testing.assert_allclose(norms, 0.866, rtol=1e-3)

    sgeom.update_areas(sheet)
    np.testing.assert_allclose(sheet.face_df["area"], np.sqrt(3) * 1.5, rtol=1e-3)

    sgeom.update_all(sheet)
    np.testing.assert_allclose(sheet.face_df["area"], np.sqrt(3) * 1.5, rtol=1e-3)


def test_face_rotation():

    h5store = os.path.join(stores_dir, "small_hexagonal.hf5")
    datasets = load_datasets(h5store, data_names=["face", "vert", "edge"])
    specs = config.geometry.cylindrical_sheet()

    sheet = Sheet("emin", datasets, specs)
    sgeom.update_all(sheet)
    face = 17
    normal = sheet.edge_df[sheet.edge_df["face"] == face][sheet.ncoords].mean()
    rot = sgeom.face_rotation(sheet, face, 0)
    rotated = np.dot(rot, normal)
    np.testing.assert_allclose(rotated[:2], np.zeros(2), atol=1e-7)


def test_center():

    sheet = Sheet.planar_sheet_3d("sheet", 5, 5, 1, 1)
    sheet.sanitize()
    sgeom.update_all(sheet)
    sgeom.center(sheet)
    np.testing.assert_allclose(
        sheet.vert_df[sheet.coords].mean(axis=0), [0, 0, 0], atol=1e-7
    )


def test_rotations():
    h5store = os.path.join(stores_dir, "small_hexagonal.hf5")
    datasets = load_datasets(h5store, data_names=["face", "vert", "edge"])
    specs = config.geometry.cylindrical_sheet()

    sheet = Sheet("emin", datasets, specs)
    sgeom.update_all(sheet)
    sheet.sanitize()
    sgeom.update_all(sheet)
    sgeom.center(sheet)
    assert (
        sgeom.face_rotations(sheet, method="normal", output_as="edge").shape[0]
        == sheet.Ne
    )
    assert (
        sgeom.face_rotations(sheet, method="normal", output_as="face").shape[0]
        == sheet.Nf
    )
    assert (
        sgeom.face_rotations(sheet, method="svd", output_as="edge").shape[0] == sheet.Ne
    )
    assert (
        sgeom.face_rotations(sheet, method="svd", output_as="face").shape[0] == sheet.Nf
    )


def _jittered(eptm, geom, seed=0):
    geom.update_all(eptm)
    rng = np.random.default_rng(seed)
    eptm.vert_df[eptm.coords] += rng.normal(0, 0.03, (eptm.Nv, len(eptm.coords)))
    geom.update_all(eptm)
    return eptm


def _open_cylinder():
    datasets = load_datasets(
        os.path.join(stores_dir, "small_hexagonal.hf5"),
        data_names=["face", "vert", "edge"],
    )
    return Sheet("cylinder", datasets, config.geometry.cylindrical_sheet())


def _deviation_brute_force(sheet):
    """Offset of each vertex from the mean of its neighbours, linked either way."""
    edges, pos = sheet.edge_df, sheet.vert_df[["x", "y", "z"]]
    rows = []
    for vert in sheet.vert_df.index:
        neighbors = np.union1d(
            edges.loc[edges["srce"] == vert, "trgt"], edges.loc[edges["trgt"] == vert, "srce"]
        )
        deviation = pos.loc[vert].to_numpy() - pos.loc[neighbors].to_numpy().mean(axis=0)
        length = np.linalg.norm(deviation)
        rows.append([length, *(deviation / length)])
    return np.array(rows)


def test_vert_deviation():
    from tyssue.geometry.cylinder_geometry import CylinderGeometryInit
    from tyssue.utils.connectivity import vertex_rank

    cols = ["dev_length", "dx", "dy", "dz"]

    # open mesh: boundary vertices of rank 2 and 3 and interior vertices alike
    sheet = _open_cylinder()
    CylinderGeometryInit.update_all(sheet)  # used to fail on border vertices
    assert set(vertex_rank(sheet)) == {2, 3}
    np.testing.assert_allclose(
        sheet.vert_df[cols].to_numpy(), _deviation_brute_force(sheet), atol=1e-12
    )

    # a rank 4 vertex: every neighbour counts, whatever the edge order
    sheet = Sheet.planar_sheet_3d("flat", 8, 8, 1, 1)
    sheet.sanitize()
    sgeom.update_all(sheet)
    sheet.vert_df["z"] += np.random.default_rng(0).normal(0, 0.05, sheet.Nv)
    inner = sheet.edge_df.index[sheet.edge_df["opposite"] >= 0]
    collapse_edge(sheet, inner[len(inner) // 2])
    sgeom.update_all(sheet)
    assert vertex_rank(sheet).max() == 4
    CylinderGeometryInit.update_vert_deviation(sheet)
    expected = sheet.vert_df[cols].to_numpy().copy()
    np.testing.assert_allclose(expected, _deviation_brute_force(sheet), atol=1e-12)

    shuffled = sheet.copy()
    shuffled.edge_df = shuffled.edge_df.sample(frac=1, random_state=0)
    CylinderGeometryInit.update_vert_deviation(shuffled)
    np.testing.assert_array_equal(shuffled.vert_df[cols].to_numpy(), expected)


def _idempotence_cases():
    from tyssue import Monolayer, Sheet, config
    from tyssue.generation import ellipsoid_sheet, extrude, three_faces_sheet
    from tyssue.generation.shapes import generate_ring
    from tyssue.geometry.bulk_geometry import BulkGeometry, MonolayerGeometry
    from tyssue.geometry.cylinder_geometry import (
        CylinderGeometry,
        CylinderGeometryInit,
    )
    from tyssue.geometry.planar_geometry import AnnularGeometry, PlanarGeometry
    from tyssue.geometry.sheet_geometry import (
        ClosedSheetGeometry,
        EllipsoidGeometry,
        SheetGeometry,
    )
    from tyssue.geometry.vessel_geometry import VesselGeometry

    def planar():
        sheet = Sheet.planar_sheet_2d("p", 8, 8, 1, 1)
        sheet.sanitize()
        return sheet

    def flat():
        sheet = Sheet.planar_sheet_3d("f", 8, 8, 1, 1)
        sheet.sanitize()
        return sheet

    def mono():
        data, _ = three_faces_sheet()
        return Monolayer("m", extrude(data), config.geometry.bulk_spec())

    return [
        (PlanarGeometry, planar),
        (AnnularGeometry, lambda: generate_ring(12, 1, 2)),
        (SheetGeometry, flat),
        (VesselGeometry, flat),
        (ClosedSheetGeometry, lambda: ellipsoid_sheet(3, 3, 4, 12)),
        (EllipsoidGeometry, lambda: ellipsoid_sheet(3, 3, 4, 12)),
        (CylinderGeometryInit, _open_cylinder),
        (CylinderGeometry, _open_cylinder),
        (BulkGeometry, mono),
        (MonolayerGeometry, mono),
    ]


@pytest.mark.parametrize(
    "geom, make", _idempotence_cases(), ids=lambda c: getattr(c, "__name__", "")
)
def test_update_all_idempotent(geom, make):
    """A second update_all on unchanged positions and topology changes nothing,
    which is what lets the solver skip it (`idempotent_update`)."""
    assert geom.idempotent_update
    eptm = _jittered(make(), geom)
    before = {name: df.copy() for name, df in eptm.datasets.items()}
    geom.update_all(eptm)
    for name, df in eptm.datasets.items():
        for col in df.columns:
            if df[col].dtype.kind in "fi":
                np.testing.assert_array_equal(
                    df[col].to_numpy(), before[name][col].to_numpy(), err_msg=f"{name}.{col}"
                )
    np.testing.assert_allclose(
        np.linalg.norm(eptm.edge_df[eptm.ucoords].to_numpy(), axis=1), 1.0, rtol=1e-12
    )


def test_unit_vectors_after_type1():
    """Edges created by a T1 copy their parent's row, length included; the unit
    vectors must still come out of a single update_all normalised."""
    from tyssue import Sheet
    from tyssue.geometry.sheet_geometry import SheetGeometry
    from tyssue.topology.sheet_topology import type1_transition

    sheet = Sheet.planar_sheet_3d("f", 8, 8, 1, 1)
    sheet.sanitize()
    SheetGeometry.update_all(sheet)
    sheet.settings["threshold_length"] = 0.1
    type1_transition(sheet, sheet.edge_df.index[sheet.edge_df["opposite"] >= 0][5])
    SheetGeometry.update_all(sheet)
    np.testing.assert_allclose(
        np.linalg.norm(sheet.edge_df[sheet.ucoords].to_numpy(), axis=1), 1.0, rtol=1e-12
    )
