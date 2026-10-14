import numpy as np
import pandas as pd
from numpy import pi

import pytest

from tyssue import config
from tyssue.core import Epithelium
from tyssue.geometry.planar_geometry import PlanarGeometry


def test_face_projected_pos():
    datasets = {}
    tri_verts = [[0, 0], [1, 0], [-0.5, 0.75], [-0.5, -0.75]]

    tri_edges = [
        [0, 1, 0],
        [1, 2, 0],
        [2, 0, 0],
        [0, 3, 1],
        [3, 1, 1],
        [1, 0, 1],
        [0, 2, 2],
        [2, 3, 2],
        [3, 0, 2],
    ]

    datasets["edge"] = pd.DataFrame(
        data=np.array(tri_edges), columns=["srce", "trgt", "face"]
    )
    datasets["edge"].index.name = "edge"
    datasets["face"] = pd.DataFrame(data=np.zeros((3, 2)), columns=["x", "y"])
    datasets["face"].index.name = "face"

    datasets["vert"] = pd.DataFrame(data=np.array(tri_verts), columns=["x", "y"])
    datasets["vert"].index.name = "vert"
    specs = config.geometry.planar_spec()
    eptm = Epithelium("extra", datasets, specs, coords=["x", "y"])
    PlanarGeometry.update_all(eptm)

    res_rot_pos_pi2 = PlanarGeometry.face_projected_pos(eptm, 0, pi / 2.0)
    res_rot_pos_face1_2pi = PlanarGeometry.face_projected_pos(eptm, 1, 2.0 * pi)

    expected_rot_pos_pi2 = pd.DataFrame.from_dict(
        {
            "vert": [0, 1, 2, 3],
            "x": [0.25, 0.25, -0.5, 1.0],
            "y": [-0.166667, 0.8333333, -0.666667, -0.666667],
        }
    ).set_index("vert")

    expected_rot_pos_face1_2pi = pd.DataFrame.from_dict(
        {
            "vert": [0, 1, 2, 3],
            "x": [-0.166667, 0.833333, -0.666667, -0.666667],
            "y": [0.25, 0.25, 1.00, -0.5],
        }
    ).set_index("vert")
    tolerance = 1e-16
    assert all((expected_rot_pos_pi2 - res_rot_pos_pi2) ** 2 < tolerance)
    assert all((expected_rot_pos_face1_2pi - res_rot_pos_face1_2pi) ** 2 < tolerance)


@pytest.mark.skip("waith till we sort out collisions")
def test_update_repulsion():
    datasets = {}
    tri_verts = [[0, 0], [1, 0], [-0.5, 0.75], [-0.5, -0.75]]

    tri_edges = [
        [0, 1, 0],
        [1, 2, 0],
        [2, 0, 0],
        [0, 3, 1],
        [3, 1, 1],
        [1, 0, 1],
        [0, 2, 2],
        [2, 3, 2],
        [3, 0, 2],
    ]

    datasets["edge"] = pd.DataFrame(
        data=np.array(tri_edges), columns=["srce", "trgt", "face"]
    )
    datasets["edge"].index.name = "edge"
    datasets["face"] = pd.DataFrame(data=np.zeros((3, 2)), columns=["x", "y"])
    datasets["face"].index.name = "face"

    datasets["vert"] = pd.DataFrame(data=np.array(tri_verts), columns=["x", "y"])
    datasets["vert"].index.name = "vert"
    specs = config.geometry.planar_spec()
    specs["vert"]["force_repulsion"] = 10
    eptm = Epithelium("extra", datasets, specs, coords=["x", "y"])
    PlanarGeometry.update_all(eptm)
    PlanarGeometry.update_repulsion(eptm)

    assert "grid" in eptm.vert_df.columns
    assert "v_repulsion" in eptm.vert_df.columns
    assert len(np.unique(eptm.vert_df.loc[1, "v_repulsion"])) > 1


def _legacy_boundary_index(sheet):
    """The boundary index as computed before it was cached (reference)."""
    from tyssue.core.sheet import get_opposite

    sheet.vert_df["boundary"] = 0
    sheet.edge_df["boundary"] = 0
    sheet.face_df["boundary"] = 0
    sheet.edge_df["opposite"] = get_opposite(sheet.edge_df)
    free = sheet.edge_df["opposite"] == -1
    sheet.edge_df.loc[free, "boundary"] = 1
    sheet.vert_df.loc[sheet.edge_df.loc[free, "trgt"].unique(), "boundary"] = 1
    faces = sheet.edge_df.loc[free, "face"].dropna().unique().astype(int)
    sheet.face_df.loc[faces, "boundary"] = 1


def _boundary_meshes():
    from tyssue import Sheet
    from tyssue.generation import ellipsoid_sheet, three_faces_sheet

    planar = Sheet.planar_sheet_2d("planar", 8, 8, 1, 1)
    planar.sanitize()
    flat = Sheet.planar_sheet_3d("flat", 8, 8, 1, 1)
    flat.sanitize()
    shuffled = flat.copy()
    shuffled.vert_df = shuffled.vert_df.sample(frac=1, random_state=0)
    return {
        "planar_2d": planar,
        "planar_3d": flat,
        "shuffled_verts": shuffled,
        "three_faces": Sheet("3f", *three_faces_sheet()),
        "ellipsoid": ellipsoid_sheet(3, 3, 4, 12),
    }


@pytest.mark.parametrize("name", list(_boundary_meshes()))
def test_update_boundary_index_matches_legacy(name):
    sheet = _boundary_meshes()[name]
    ref = sheet.copy()
    _legacy_boundary_index(ref)
    PlanarGeometry.update_boundary_index(sheet)
    for df in ("vert_df", "edge_df", "face_df"):
        got, exp = getattr(sheet, df)["boundary"], getattr(ref, df)["boundary"]
        assert got.dtype == exp.dtype
        np.testing.assert_array_equal(got.to_numpy(), exp.to_numpy())
    np.testing.assert_array_equal(
        sheet.edge_df["opposite"].to_numpy(), ref.edge_df["opposite"].to_numpy()
    )
    bnd = sheet._boundary
    np.testing.assert_array_equal(
        bnd.free_edges, np.flatnonzero(ref.edge_df["opposite"].to_numpy() == -1)
    )
    np.testing.assert_array_equal(
        bnd.verts, np.flatnonzero(ref.vert_df["boundary"].to_numpy() == 1)
    )


def test_planar_update_all_sets_boundary():
    from tyssue import Sheet

    sheet = Sheet.planar_sheet_2d("planar", 6, 6, 1, 1)
    sheet.sanitize()
    PlanarGeometry.update_all(sheet)
    for df in (sheet.vert_df, sheet.edge_df, sheet.face_df):
        assert "boundary" in df.columns
    assert sheet.edge_df["boundary"].sum() == (sheet.edge_df["opposite"] == -1).sum()


def _count_get_opposite(monkeypatch):
    import tyssue.core.sheet

    calls = []
    original = tyssue.core.sheet.get_opposite

    def counting(edge_df, *args, **kwargs):
        calls.append(1)
        return original(edge_df, *args, **kwargs)

    monkeypatch.setattr(tyssue.core.sheet, "get_opposite", counting)
    return calls


def test_boundary_index_is_cached(monkeypatch):
    from tyssue import Sheet

    sheet = Sheet.planar_sheet_2d("planar", 6, 6, 1, 1)
    sheet.sanitize()
    calls = _count_get_opposite(monkeypatch)
    PlanarGeometry.update_boundary_index(sheet)
    assert len(calls) == 1
    sheet.vert_df["x"] += 0.1  # geometry only
    PlanarGeometry.update_all(sheet)
    PlanarGeometry.update_boundary_index(sheet)
    assert len(calls) == 1

    # a dropped column is restored
    sheet.vert_df.drop(columns="boundary", inplace=True)
    PlanarGeometry.update_boundary_index(sheet)
    assert len(calls) == 2
    assert "boundary" in sheet.vert_df.columns


def test_boundary_index_follows_topology(monkeypatch):
    from tyssue import Sheet

    sheet = Sheet.planar_sheet_2d("planar", 6, 6, 1, 1)
    sheet.sanitize()
    PlanarGeometry.update_boundary_index(sheet)
    calls = _count_get_opposite(monkeypatch)

    # reverse a border face: same element counts, new connectivity (as a T1)
    face = sheet.edge_df.loc[sheet.edge_df["boundary"] == 1, "face"].iloc[0]
    rows = sheet.edge_df["face"] == face
    srce = sheet.edge_df.loc[rows, "srce"].to_numpy()
    sheet.edge_df.loc[rows, "srce"] = sheet.edge_df.loc[rows, "trgt"].to_numpy()
    sheet.edge_df.loc[rows, "trgt"] = srce
    PlanarGeometry.update_boundary_index(sheet)
    assert len(calls) == 1
    ref = sheet.copy()
    _legacy_boundary_index(ref)
    np.testing.assert_array_equal(
        sheet.edge_df["boundary"].to_numpy(), ref.edge_df["boundary"].to_numpy()
    )

    # a dataframe replaced wholesale
    calls.clear()
    sheet.edge_df = sheet.edge_df.copy()
    PlanarGeometry.update_boundary_index(sheet)
    assert len(calls) == 1


def test_cylinder_boundary_index_keeps_face_column():
    from tyssue import Sheet
    from tyssue.geometry.cylinder_geometry import CylinderGeometryInit

    sheet = Sheet.planar_sheet_3d("flat", 6, 6, 1, 1)
    sheet.sanitize()
    sheet.face_df["boundary"] = 7
    CylinderGeometryInit.update_boundary_index(sheet)
    assert (sheet.face_df["boundary"] == 7).all()
    assert sheet.vert_df["boundary"].sum() > 0
    # a later full index does write the face column
    PlanarGeometry.update_boundary_index(sheet)
    assert set(sheet.face_df["boundary"].unique()) == {0, 1}
