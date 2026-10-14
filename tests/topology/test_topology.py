import os

import numpy as np
import pandas as pd
import pytest

from tyssue.config.geometry import cylindrical_sheet
from tyssue.core.sheet import Sheet
from tyssue.generation import three_faces_sheet
from tyssue.geometry.planar_geometry import PlanarGeometry
from tyssue.geometry.sheet_geometry import SheetGeometry as geom
from tyssue.io.hdf5 import load_datasets
from tyssue.stores import stores_dir
from tyssue.topology.base_topology import (
    add_vert,
    close_face,
    condition_4i,
    condition_4ii,
    remove_face,
)
from tyssue.topology.sheet_topology import (
    cell_division,
    drop_face,
    split_vert,
    t3_transition,
    type1_transition,
)


def test_condition4i():
    sheet = Sheet("test", *three_faces_sheet())
    assert len(condition_4i(sheet)) == 0

    sheet.edge_df = pd.concat([sheet.edge_df, sheet.edge_df.iloc[-1:]])
    sheet.edge_df.index.name = "edge"
    sheet.reset_index()
    sheet.reset_topo()
    assert len(condition_4i(sheet)) == 1


def test_condition4ii():
    sheet = Sheet("test", *three_faces_sheet())
    assert len(condition_4ii(sheet)) == 0
    add_vert(sheet, 0)
    sheet.reset_index()
    sheet.reset_topo()
    assert len(condition_4ii(sheet)) == 2


def test_division():
    h5store = os.path.join(stores_dir, "small_hexagonal.hf5")

    datasets = load_datasets(h5store, data_names=["face", "vert", "edge"])
    specs = cylindrical_sheet()
    sheet = Sheet("emin", datasets, specs)
    geom.update_all(sheet)

    Nf, Ne, Nv = sheet.Nf, sheet.Ne, sheet.Nv

    cell_division(sheet, 17, geom)

    assert sheet.Nf - Nf == 1
    assert sheet.Nv - Nv == 2
    assert sheet.Ne - Ne == 6


def test_t1_transition():
    h5store = os.path.join(stores_dir, "small_hexagonal.hf5")
    datasets = load_datasets(h5store, data_names=["face", "vert", "edge"])
    specs = cylindrical_sheet()
    sheet = Sheet("emin", datasets, specs)
    geom.update_all(sheet)
    face = sheet.edge_df.loc[84, "face"]
    srce = sheet.edge_df.loc[84, "srce"]
    type1_transition(sheet, 84)
    assert sheet.face_df.loc[face, "num_sides"] == 5
    assert sheet.edge_df[(sheet.edge_df.srce == srce)].shape[0] == 3


def test_t1_at_border():
    datasets, specs = three_faces_sheet()
    sheet = Sheet("3cells_2D", datasets, specs)
    geom.update_all(sheet)
    # double half edge with no right cell (aka cell c)
    type1_transition(sheet, 0, multiplier=4.0)
    sheet.reset_index()
    assert sheet.validate()
    geom.update_all(sheet)
    # single half edge with no bottom cell (aka cell d)
    type1_transition(sheet, 16, multiplier=5.0)
    geom.update_all(sheet)
    assert sheet.validate()
    # single half edge with no left cell (aka cell a)
    type1_transition(sheet, 17, multiplier=5.0)
    geom.update_all(sheet)
    assert sheet.validate()


def test_split_vert():
    datasets, specs = three_faces_sheet()
    sheet = Sheet("3cells_2D", datasets, specs)
    geom.update_all(sheet)

    split_vert(sheet, 0, multiplier=10)
    geom.update_all(sheet)
    assert sheet.Nv == 14
    assert sheet.Ne == 20


def test_close_face():
    sheet = Sheet("test", *three_faces_sheet())
    e0 = sheet.edge_df.index[0]
    face = sheet.edge_df.loc[e0, "face"]
    Ne = sheet.Ne
    sheet.edge_df = sheet.edge_df.loc[sheet.edge_df.index[1:]].copy()
    close_face(sheet, face)
    assert sheet.Ne == Ne

    close_face(sheet, face)
    assert sheet.Ne == Ne


def test_remove_face():
    sheet = Sheet.planar_sheet_2d("flat", 6, 7, 1, 1, noise=1e-6)
    geom = PlanarGeometry

    to_cut = sheet.cut_out([(0.1, 6), (0.1, 6.0)])
    sheet.remove(to_cut, trim_borders=True)
    sheet.sanitize(trim_borders=True)

    geom.center(sheet)
    geom.update_all(sheet)

    remove_face(sheet, 13)
    geom.update_all(sheet)

    assert np.all(np.isfinite(sheet.vert_df.iloc[-1]))


def test_merge_border_edges():
    sheet = Sheet.planar_sheet_3d("sheet", 5, 6, 1, 1)
    sheet.get_opposite()
    sheet.sanitize(trim_borders=True)
    assert (
            sheet.edge_df[sheet.edge_df["opposite"] < 0]
            .groupby("face")["opposite"]
            .sum()
            .min()
            == -1
    )
    assert not set(sheet.vert_df.index).difference(sheet.edge_df.srce)


def test_drop_face():
    init_sheet = Sheet.planar_sheet_2d("planar", 6, 7, 1, 1)
    init_sheet.sanitize(trim_borders=True, order_edges=True)
    geom = PlanarGeometry
    sheet = init_sheet.copy(deep_copy=True)

    drop_face(sheet, 12, geom)
    drop_face(sheet, 11, geom)

    assert sheet.Nf == init_sheet.Nf - 2
    assert sheet.Nv == init_sheet.Nv - 2
    assert sheet.Ne == init_sheet.Ne - 14

    sheet.get_opposite()
    init_sheet.get_opposite()
    assert sheet.edge_df[sheet.edge_df['opposite'] == -1].shape[0] == \
           init_sheet.edge_df[init_sheet.edge_df['opposite'] == -1].shape[0] + 8


def _square_triangle(gap=0.005):
    """A square and a triangle whose tip is `gap` away from the square's right
    (free) edge, the half-edge 1 -> 2."""
    from tyssue.config.geometry import planar_spec

    vert = pd.DataFrame(
        {"x": [0.0, 1, 1, 0, 1 + gap, 2, 2], "y": [0.0, 0, 1, 1, 0.5, -0.5, 1.5]}
    )
    edge = pd.DataFrame(
        {
            "srce": [0, 1, 2, 3, 4, 5, 6],
            "trgt": [1, 2, 3, 0, 5, 6, 4],
            "face": [0, 0, 0, 0, 1, 1, 1],
        }
    )
    face = pd.DataFrame({"x": [0.5, 1.7], "y": [0.5, 0.5]})
    for df, name in ((vert, "vert"), (edge, "edge"), (face, "face")):
        df.index.name = name
    sheet = Sheet(
        "sq_tri",
        {"vert": vert, "edge": edge, "face": face},
        specs=planar_spec(),
        coords=["x", "y"],
    )
    PlanarGeometry.update_all(sheet)
    return sheet


def test_t3_transition():
    sheet = _square_triangle()
    ne = sheet.Ne
    assert t3_transition(sheet, 4, 1) == 0
    PlanarGeometry.update_all(sheet)
    assert sheet.validate()
    assert sheet.Ne == ne + 1
    assert sheet.face_df.loc[0, "num_sides"] == 5
    ring = sheet.edge_df[sheet.edge_df["face"] == 0]
    assert set(ring["srce"]) == {0, 1, 2, 3, 4}
    # snapped on the edge, and the border stays free on both sides
    np.testing.assert_allclose(sheet.vert_df.loc[4, ["x", "y"]], [1.0, 0.5])
    assert (sheet.edge_df["opposite"] == -1).all()
    assert sheet.edge_df["unique_id"].is_unique


def test_t3_transition_no_snap():
    sheet = _square_triangle()
    assert t3_transition(sheet, 4, 1, snap=False) == 0
    np.testing.assert_allclose(sheet.vert_df.loc[4, ["x", "y"]], [1.005, 0.5])


def test_t3_transition_refused():
    sheet = _square_triangle()
    # its own end
    assert t3_transition(sheet, 2, 1) == -1
    # a vertex of the same face
    assert t3_transition(sheet, 0, 1) == -1
    # linked to an end of the edge by v -> b: a second v -> b would be created
    sheet.edge_df.loc[5, "trgt"] = 2
    sheet.edge_df.loc[6, "srce"] = 2
    assert t3_transition(sheet, 5, 1) == -1

    # an edge with an opposite is not on the border
    sheet = Sheet("3f", *three_faces_sheet())
    geom.update_all(sheet)
    inner = sheet.edge_df[sheet.edge_df["opposite"] >= 0].index[0]
    face = sheet.edge_df.loc[inner, "face"]
    other = sheet.edge_df.loc[
        ~sheet.edge_df["srce"].isin(sheet.edge_df.loc[sheet.edge_df["face"] == face, "srce"]),
        "srce",
    ].iloc[0]
    ne = sheet.Ne
    assert t3_transition(sheet, other, inner) == -1
    assert sheet.Ne == ne


def _square_wedge(gap=0.005):
    """A square and a triangle sharing the square's corner 2, the triangle's
    free edge 2 -> 4 running down along the square's right (free) edge 1 -> 2,
    with vertex 4 `gap` away from it."""
    from tyssue.config.geometry import planar_spec

    vert = pd.DataFrame(
        {"x": [0.0, 1, 1, 0, 1 + gap, 1.8], "y": [0.0, 0, 1, 1, 0.4, 0.5]}
    )
    edge = pd.DataFrame(
        {
            "srce": [0, 1, 2, 3, 2, 4, 5],
            "trgt": [1, 2, 3, 0, 4, 5, 2],
            "face": [0, 0, 0, 0, 1, 1, 1],
        }
    )
    face = pd.DataFrame({"x": [0.5, 1.3], "y": [0.5, 0.6]})
    for df, name in ((vert, "vert"), (edge, "edge"), (face, "face")):
        df.index.name = name
    sheet = Sheet(
        "sq_wedge",
        {"vert": vert, "edge": edge, "face": face},
        specs=planar_spec(),
        coords=["x", "y"],
    )
    PlanarGeometry.update_all(sheet)
    return sheet


def test_t3_transition_zips_linked_free_edges():
    # 4 is linked to the end 2 of the edge 1 -> 2 by the triangle's 2 -> 4
    sheet = _square_wedge()
    assert (sheet.edge_df["opposite"] == -1).all()
    ne = sheet.Ne
    assert t3_transition(sheet, 4, 1) == 0
    PlanarGeometry.update_all(sheet)
    assert sheet.validate()
    assert sheet.Ne == ne + 1
    # the square's new 4 -> 2 is the opposite of the triangle's 2 -> 4
    edge_df = sheet.edge_df
    shared = edge_df[edge_df["opposite"] >= 0]
    assert sorted(zip(shared["srce"], shared["trgt"], shared["face"])) == [
        (2, 4, 1),
        (4, 2, 0),
    ]
    np.testing.assert_allclose(sheet.vert_df.loc[4, ["x", "y"]], [1.0, 0.4])


def test_t3_transition_refuses_duplicate_half_edge():
    # the same wedge with the triangle turned the other way: 4 -> 2 is a
    # half-edge of the triangle, and the square would need a second one
    sheet = _square_wedge()
    sheet.edge_df[["srce", "trgt"]] = sheet.edge_df[["trgt", "srce"]].to_numpy()
    sheet.edge_df.loc[[0, 1, 2, 3], ["srce", "trgt"]] = [[0, 1], [1, 2], [2, 3], [3, 0]]
    ne = sheet.Ne
    assert t3_transition(sheet, 4, 1) == -1
    assert sheet.Ne == ne


def test_single_rearangement_with_t3_alias():
    from tyssue.topology import single_rearangement

    sheet = Sheet("3f", *three_faces_sheet())
    geom.update_all(sheet)
    with pytest.warns(FutureWarning, match="with_t2"):
        single_rearangement(sheet, with_t3=True)


def test_auto_t3_alias():
    from tyssue.topology import auto_t2, auto_t3

    def fun(eptm, geom):
        return 1

    with pytest.warns(FutureWarning, match="auto_t2"):
        wrapped = auto_t3(fun)
    assert wrapped.__wrapped__ is fun
    assert auto_t2(fun).__wrapped__ is fun
