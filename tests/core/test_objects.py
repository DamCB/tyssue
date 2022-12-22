import os

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pytest import raises

from tyssue import PlanarGeometry, config
from tyssue.core import Epithelium
from tyssue.core.objects import (
    _ordered_edges,
    _ordered_vert_idxs,
    get_next_edges,
    get_prev_edges,
    get_simple_index,
)
from tyssue.core.sheet import Sheet, get_opposite
from tyssue.dynamics import effectors, model_factory
from tyssue.generation import extrude, three_faces_sheet
from tyssue.geometry.bulk_geometry import RNRGeometry
from tyssue.geometry.sheet_geometry import SheetGeometry
from tyssue.io.hdf5 import load_datasets
from tyssue.solvers.quasistatic import QSSolver
from tyssue.stores import stores_dir


def test_3faces():

    datasets, specs = three_faces_sheet()
    eptm = Epithelium("3faces_2D", datasets, specs)
    assert (eptm.Nc, eptm.Nv, eptm.Ne) == (3, 13, 18)


def test_idx_lookup():
    datasets, specs = three_faces_sheet()
    eptm = Epithelium("3faces_2D", datasets, specs)
    eptm.face_df["id"] = eptm.face_df.index.values
    assert eptm.idx_lookup(1, "face") == 1


def test_triangular_mesh():
    datasets, specs = three_faces_sheet()
    eptm = Epithelium("3faces_2D", datasets, specs)
    vertices, faces = eptm.triangular_mesh(["x", "y", "z"], False)
    assert vertices.shape == (16, 3)
    assert faces.shape == (18, 3)


def test_opposite():
    datasets, _ = three_faces_sheet()
    opposites = get_opposite(datasets["edge"])
    true_opp = np.array(
        [
            17.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            6.0,
            5.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            12.0,
            11.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            0.0,
        ]
    )
    assert_array_equal(true_opp, opposites)

    edge_df = datasets["edge"].append(datasets["edge"].loc[0], ignore_index=True)
    edge_df.index.name = "edge"
    with pytest.warns(UserWarning):
        opposites = get_opposite(edge_df)
        true_opp = np.array(
            [
                17.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                6.0,
                5.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                12.0,
                11.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                0.0,
                17,
            ]
        )
        assert_array_equal(true_opp, opposites)


def test_extra_indices():

    datasets = {}
    tri_verts = [[0, 0], [1, 0], [-0.5, 3**0.5 / 2], [-0.5, -(3**0.5) / 2]]

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
    eptm = Sheet("extra", datasets, specs, coords=["x", "y"])
    PlanarGeometry.update_all(eptm)
    eptm.edge_df["opposite"] = get_opposite(eptm.edge_df)
    eptm.get_extra_indices()

    assert (2 * eptm.Ni + eptm.No) == eptm.Ne
    assert eptm.west_edges.size == eptm.Ni
    assert eptm.Nd == 2 * eptm.Ni

    for edge in eptm.free_edges:
        opps = eptm.edge_df[eptm.edge_df["srce"] == eptm.edge_df.loc[edge, "trgt"]][
            "trgt"
        ]
        assert eptm.edge_df.loc[edge, "srce"] not in opps

    for edge in eptm.east_edges:
        srce, trgt = eptm.edge_df.loc[edge, ["srce", "trgt"]]
        opp = eptm.edge_df[
            (eptm.edge_df["srce"] == trgt) & (eptm.edge_df["trgt"] == srce)
        ].index
        assert opp[0] in eptm.west_edges

    for edge in eptm.west_edges:
        srce, trgt = eptm.edge_df.loc[edge, ["srce", "trgt"]]
        opp = eptm.edge_df[
            (eptm.edge_df["srce"] == trgt) & (eptm.edge_df["trgt"] == srce)
        ].index
        assert opp[0] in eptm.east_edges


def test_extra_indices_hexabug():
    # GH #192

    with pytest.raises(AssertionError):
        h5store = os.path.join(stores_dir, "small_hexagonal_snaped.hf5")

        datasets = load_datasets(h5store)
        specs = config.geometry.cylindrical_sheet()
        sheet = Sheet("emin", datasets, specs)

        SheetGeometry.update_all(sheet)
        sheet.sanitize()
        sheet.get_extra_indices()

    h5store = os.path.join(stores_dir, "small_hexagonal.hf5")

    datasets = load_datasets(h5store)
    specs = config.geometry.cylindrical_sheet()
    sheet = Sheet("emin", datasets, specs)

    SheetGeometry.update_all(sheet)
    sheet.sanitize()
    sheet.get_extra_indices()


def test_sort_eastwest():
    datasets = {}
    tri_verts = [[0, 0], [1, 0], [-0.5, 3**0.5 / 2], [-0.5, -(3**0.5) / 2]]

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
    eptm = Sheet("extra", datasets, specs, coords=["x", "y"])
    PlanarGeometry.update_all(eptm)
    eptm.edge_df["opposite"] = get_opposite(eptm.edge_df)
    eptm.sort_edges_eastwest()
    assert_array_equal(np.asarray(eptm.free_edges), [0, 1, 2])
    assert_array_equal(np.asarray(eptm.east_edges), [3, 4, 5])
    assert_array_equal(np.asarray(eptm.west_edges), [6, 7, 8])


def test_update_rank():

    sheet = Sheet("3", *three_faces_sheet())
    sheet.update_rank()
    np.testing.assert_array_equal(
        np.array([3, 3, 2, 2, 2, 3, 2, 2, 2, 3, 2, 2, 2]), sheet.vert_df["rank"]
    )

    mono = Epithelium("3", extrude(sheet.datasets))
    mono.update_rank()
    mono.vert_df["rank"].values
    np.testing.assert_array_equal(
        np.array(
            [
                4,
                4,
                3,
                3,
                3,
                4,
                3,
                3,
                3,
                4,
                3,
                3,
                3,
                4,
                4,
                3,
                3,
                3,
                4,
                3,
                3,
                3,
                4,
                3,
                3,
                3,
            ]
        ),
        mono.vert_df["rank"].values,
    )


# BC ####


def test_wrong_datasets_keys():
    datasets, specs = three_faces_sheet()
    datasets["edges"] = datasets["edge"]
    del datasets["edge"]
    with raises(ValueError):
        Epithelium("3faces_2D", datasets, specs)


def test_optional_args_eptm():
    datasets, _ = three_faces_sheet()
    data_names = set(datasets.keys())
    eptm = Epithelium("3faces_2D", datasets)
    specs_names = set(eptm.specs.keys())
    assert specs_names.issuperset(data_names)
    assert "settings" in specs_names


def test_3d_eptm_cell_getter_setter():
    datasets_2d, _ = three_faces_sheet()
    datasets = extrude(datasets_2d)
    eptm = Epithelium("3faces_3D", datasets)
    assert eptm.cell_df is not None

    datasets_3d = extrude(datasets_2d, method="translation")
    eptm.cell_df = datasets_3d["cell"]
    for key in eptm.cell_df:
        assert key in datasets_3d["cell"]


def test_eptm_copy():
    datasets, _ = three_faces_sheet()
    eptm = Epithelium("3faces_2D", datasets)
    eptm_copy = eptm.copy()
    assert eptm_copy.identifier == eptm.identifier + "_copy"
    assert set(eptm_copy.datasets.keys()).issuperset(eptm.datasets.keys())
    eptm.settings["deepcopy"] = "original"
    eptm_deepcopy = eptm.copy(deep_copy=True)
    eptm_deepcopy.settings["deepcopy"] = "copy"
    assert eptm_deepcopy is not None
    assert eptm.settings["deepcopy"] != eptm_deepcopy.settings["deepcopy"]


def test_settings_getter_setter():
    datasets, _ = three_faces_sheet()
    eptm = Epithelium("3faces_2D", datasets)

    eptm.settings["settings1"] = 154
    # not validated in coverage
    # (Actually the 'settings' getter is called
    # and then the dictionary class setter
    # instead of directly the 'settings' setter.)
    # See http://stackoverflow.com/a/3137768
    assert "settings1" in eptm.settings
    assert eptm.specs["settings"]["settings1"] == 154
    assert eptm.settings["settings1"] == 154


def test_number_getters():
    datasets_2d, specs = three_faces_sheet(zaxis=True)
    datasets = extrude(datasets_2d)
    eptm = Epithelium("3faces_3D", datasets, specs)

    assert eptm.Nc == datasets["cell"].shape[0]
    assert eptm.Nv == datasets["vert"].shape[0]
    assert eptm.Nf == datasets["face"].shape[0]
    assert eptm.Ne == datasets["edge"].shape[0]


def test_upcast():
    datasets_2d, specs = three_faces_sheet(zaxis=True)
    datasets = extrude(datasets_2d)
    eptm = Epithelium("3faces_3D", datasets, specs)
    eptm.cell_df["test_data"] = eptm.cell_df.index
    eptm.face_df["test_data"] = eptm.face_df.index
    eptm.vert_df["test_data"] = eptm.vert_df.index

    assert_array_equal(
        eptm.upcast_srce(eptm.vert_df["test_data"]), eptm.edge_df["srce"]
    )
    assert_array_equal(eptm.upcast_srce("test_data"), eptm.edge_df["srce"])
    assert_array_equal(
        eptm.upcast_trgt(eptm.vert_df["test_data"]), eptm.edge_df["trgt"]
    )
    assert_array_equal(eptm.upcast_trgt("test_data"), eptm.edge_df["trgt"])

    assert_array_equal(eptm.upcast_face("test_data"), eptm.edge_df["face"])
    assert_array_equal(eptm.upcast_cell("test_data"), eptm.edge_df["cell"])


def test_upcast_ndarray():
    datasets_2d, specs = three_faces_sheet(zaxis=True)
    datasets = extrude(datasets_2d)
    eptm = Epithelium("3faces_3D", datasets, specs)
    data = np.arange(eptm.Nv * 3).reshape((eptm.Nv, 3))
    assert eptm.upcast_srce(data).shape == (eptm.Ne, 3)


def test_summation():
    datasets_2d, specs = three_faces_sheet(zaxis=True)
    datasets = extrude(datasets_2d)
    eptm = Epithelium("3faces_3D", datasets, specs)
    data = eptm.edge_df.index.values
    assert_array_equal(
        eptm.sum_cell(data).values.flatten(), np.array([1278, 1926, 2574])
    )

    sum_trgt = np.array(
        [
            462,
            302,
            88,
            97,
            106,
            248,
            142,
            151,
            160,
            356,
            196,
            205,
            214,
            501,
            340,
            107,
            116,
            125,
            286,
            161,
            170,
            179,
            394,
            215,
            224,
            233,
        ]
    )
    assert_array_equal(eptm.sum_trgt(data).values.flatten(), sum_trgt)

    sum_srce = np.array(
        [
            441,
            300,
            87,
            96,
            105,
            246,
            141,
            150,
            159,
            354,
            195,
            204,
            213,
            522,
            342,
            108,
            117,
            126,
            288,
            162,
            171,
            180,
            396,
            216,
            225,
            234,
        ]
    )
    assert_array_equal(eptm.sum_srce(data).values.flatten(), sum_srce)
    sum_face = np.array(
        [
            15,
            51,
            87,
            123,
            159,
            195,
            150,
            166,
            182,
            198,
            214,
            230,
            246,
            262,
            278,
            294,
            310,
            326,
            342,
            358,
            374,
            390,
            406,
            422,
        ]
    )
    assert_array_equal(eptm.sum_face(data).values.flatten(), sum_face)


def test_orbits():
    datasets_2d, specs = three_faces_sheet(zaxis=True)
    datasets = extrude(datasets_2d)
    eptm = Epithelium("3faces_3D", datasets, specs)

    expected_res_cell = datasets["edge"].groupby("srce").apply(lambda df: df["cell"])
    expected_res_face = datasets["edge"].groupby("face").apply(lambda df: df["trgt"])
    assert_array_equal(expected_res_cell, eptm.get_orbits("srce", "cell"))
    assert_array_equal(expected_res_face, eptm.get_orbits("face", "trgt"))


def test_polygons():
    datasets_2d, specs = three_faces_sheet(zaxis=True)
    datasets = extrude(datasets_2d, scale=1 / 3.0)
    eptm = Epithelium("3faces_3D", datasets, specs)
    RNRGeometry.update_all(eptm)
    with raises(ValueError):
        eptm.face_polygons()

    eptm.reset_index(order=True)
    res = eptm.face_polygons(["x", "y", "z"])
    shapes = res.apply(lambda s: s.shape in ((6, 3), (4, 3)))
    assert all(shapes)


def test_face_polygons_exception():

    datasets = {}
    tri_verts = [[0, 0], [1, 0], [-0.5, 3**0.5 / 2], [-0.5, -(3**0.5) / 2]]

    tri_edges_valid = [
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
        data=np.array(tri_edges_valid), columns=["srce", "trgt", "face"]
    )
    datasets["edge"].index.name = "edge"

    datasets["face"] = pd.DataFrame(data=np.zeros((3, 2)), columns=["x", "y"])
    datasets["face"].index.name = "face"

    datasets["vert"] = pd.DataFrame(data=np.array(tri_verts), columns=["x", "y"])
    datasets["vert"].index.name = "vert"

    specs = config.geometry.planar_spec()
    eptm = Epithelium("valid", datasets, specs, coords=["x", "y"])
    PlanarGeometry.update_all(eptm)
    eptm.reset_index(order=True)
    eptm.face_polygons(["x", "y"])


def test_invalid_valid_sanitize():
    # get_invalid and get_valid

    datasets = {}
    tri_verts = [[0, 0], [1, 0], [-0.5, 3**0.5 / 2], [-0.5, -(3**0.5) / 2]]

    tri_edges_valid = [
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

    tri_edges_invalid = [
        [0, 1, 0],
        [1, 2, 0],
        [2, 0, 0],
        [0, 3, 1],
        [3, 1, 1],
        [1, 0, 1],
        [0, 2, 2],
        [2, 3, 2],
        [3, 1, 2],
    ]  # changed 0 to 1 to create an invalid face

    # Epithelium whose faces are all valid
    ##
    datasets["edge"] = pd.DataFrame(
        data=np.array(tri_edges_valid), columns=["srce", "trgt", "face"]
    )
    datasets["edge"].index.name = "edge"

    datasets["face"] = pd.DataFrame(data=np.zeros((3, 2)), columns=["x", "y"])
    datasets["face"].index.name = "face"

    datasets["vert"] = pd.DataFrame(data=np.array(tri_verts), columns=["x", "y"])
    datasets["vert"].index.name = "vert"

    specs = config.geometry.planar_spec()
    eptm = Epithelium("valid", datasets, specs, coords=["x", "y"])
    PlanarGeometry.update_all(eptm)

    # Epithelium with invalid faces (last 3)
    datasets_invalid = datasets.copy()
    datasets_invalid["edge"] = pd.DataFrame(
        data=np.array(tri_edges_invalid), columns=["srce", "trgt", "face"]
    )
    datasets_invalid["edge"].index.name = "edge"

    eptm_invalid = Epithelium("invalid", datasets_invalid, specs, coords=["x", "y"])
    PlanarGeometry.update_all(eptm_invalid)

    eptm.get_valid()
    eptm_invalid.get_valid()

    res_invalid_expect_all_false = eptm.get_invalid()
    res_invalid_expect_some_true = eptm_invalid.get_invalid()

    assert eptm.edge_df["is_valid"].all()
    assert not eptm_invalid.edge_df["is_valid"][6:].all()
    assert not res_invalid_expect_all_false.all()
    assert res_invalid_expect_some_true[6:].all()

    # testing sanitize
    # edges 6 to 9 should be removed.
    edge_df_before = eptm.edge_df.copy()
    edge_df_invalid_before = eptm_invalid.edge_df.copy()

    eptm.sanitize()
    eptm_invalid.sanitize()

    assert edge_df_before.equals(eptm.edge_df)
    assert edge_df_invalid_before[:6].equals(eptm_invalid.edge_df)


def test_remove():
    datasets, specs = three_faces_sheet()
    datasets = extrude(datasets, method="translation")

    eptm = Epithelium("3Faces_3D", datasets, specs)

    dict_after = {
        "srce": {
            0: 0,
            1: 2,
            2: 3,
            3: 4,
            4: 5,
            5: 6,
            6: 0,
            7: 6,
            8: 7,
            9: 8,
            10: 9,
            11: 1,
            12: 12,
            13: 13,
            14: 14,
            15: 15,
            16: 16,
            17: 10,
            18: 16,
            19: 17,
            20: 18,
            21: 19,
            22: 11,
            23: 10,
            24: 2,
            25: 0,
            26: 10,
            27: 12,
            28: 3,
            29: 2,
            30: 12,
            31: 13,
            32: 4,
            33: 3,
            34: 13,
            35: 14,
            36: 5,
            37: 4,
            38: 14,
            39: 15,
            40: 6,
            41: 5,
            42: 15,
            43: 16,
            44: 0,
            45: 6,
            46: 16,
            47: 10,
            48: 6,
            49: 0,
            50: 10,
            51: 16,
            52: 7,
            53: 6,
            54: 16,
            55: 17,
            56: 8,
            57: 7,
            58: 17,
            59: 18,
            60: 9,
            61: 8,
            62: 18,
            63: 19,
            64: 1,
            65: 9,
            66: 19,
            67: 11,
            68: 0,
            69: 1,
            70: 11,
            71: 10,
        },
        "trgt": {
            0: 2,
            1: 3,
            2: 4,
            3: 5,
            4: 6,
            5: 0,
            6: 6,
            7: 7,
            8: 8,
            9: 9,
            10: 1,
            11: 0,
            12: 10,
            13: 12,
            14: 13,
            15: 14,
            16: 15,
            17: 16,
            18: 10,
            19: 16,
            20: 17,
            21: 18,
            22: 19,
            23: 11,
            24: 0,
            25: 10,
            26: 12,
            27: 2,
            28: 2,
            29: 12,
            30: 13,
            31: 3,
            32: 3,
            33: 13,
            34: 14,
            35: 4,
            36: 4,
            37: 14,
            38: 15,
            39: 5,
            40: 5,
            41: 15,
            42: 16,
            43: 6,
            44: 6,
            45: 16,
            46: 10,
            47: 0,
            48: 0,
            49: 10,
            50: 16,
            51: 6,
            52: 6,
            53: 16,
            54: 17,
            55: 7,
            56: 7,
            57: 17,
            58: 18,
            59: 8,
            60: 8,
            61: 18,
            62: 19,
            63: 9,
            64: 9,
            65: 19,
            66: 11,
            67: 1,
            68: 1,
            69: 11,
            70: 10,
            71: 0,
        },
        "face": {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 1,
            7: 1,
            8: 1,
            9: 1,
            10: 1,
            11: 1,
            12: 2,
            13: 2,
            14: 2,
            15: 2,
            16: 2,
            17: 2,
            18: 3,
            19: 3,
            20: 3,
            21: 3,
            22: 3,
            23: 3,
            24: 4,
            25: 4,
            26: 4,
            27: 4,
            28: 5,
            29: 5,
            30: 5,
            31: 5,
            32: 6,
            33: 6,
            34: 6,
            35: 6,
            36: 7,
            37: 7,
            38: 7,
            39: 7,
            40: 8,
            41: 8,
            42: 8,
            43: 8,
            44: 9,
            45: 9,
            46: 9,
            47: 9,
            48: 10,
            49: 10,
            50: 10,
            51: 10,
            52: 11,
            53: 11,
            54: 11,
            55: 11,
            56: 12,
            57: 12,
            58: 12,
            59: 12,
            60: 13,
            61: 13,
            62: 13,
            63: 13,
            64: 14,
            65: 14,
            66: 14,
            67: 14,
            68: 15,
            69: 15,
            70: 15,
            71: 15,
        },
    }

    sft_after = pd.DataFrame.from_dict(dict_after)

    eptm.remove([0])

    assert eptm.edge_df[["srce", "trgt", "face"]].equals(
        sft_after[["srce", "trgt", "face"]]
    )


def test_cut_out():
    datasets_2d, _ = three_faces_sheet()
    datasets = extrude(datasets_2d, method="translation")

    eptm = Epithelium("3faces_3D", datasets)

    bounding_box_xy = np.array([[-1.0, 10.0], [-1.5, 1.5]])
    bounding_box_yx = np.array([[-1.5, 1.5], [-1.0, 10.0]])
    bounding_box_xyz = np.array([[-10.0, 10.0], [-1.5, 10.0], [-2.0, 1.0]])

    expected_index_xy = pd.Index(
        [
            2,
            3,
            4,
            7,
            8,
            9,
            10,
            13,
            14,
            15,
            20,
            21,
            22,
            25,
            26,
            27,
            28,
            31,
            32,
            33,
            44,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            64,
            66,
            67,
            68,
            69,
            70,
            71,
            72,
            73,
            74,
            75,
            76,
            77,
            78,
            88,
            90,
            91,
            92,
            93,
            94,
            95,
            96,
            97,
            98,
        ],
        name="edge",
        dtype="int64",
    )

    expected_index_xyz = pd.Index(
        [13, 14, 15, 31, 32, 33, 88, 90, 91, 92, 93, 94, 95, 96, 97, 98],
        name="edge",
        dtype="int64",
    )

    # test 2-coords, ordered
    res = eptm.cut_out(bbox=bounding_box_xy, coords=["x", "y"])
    assert len(res) == len(expected_index_xy)
    assert (res == expected_index_xy).all()

    # test 2-coords, inverse order
    res = eptm.cut_out(bbox=bounding_box_yx, coords=["y", "x"])
    assert len(res) == len(expected_index_xy)
    assert (res == expected_index_xy).all()

    # test 3-coords
    res = eptm.cut_out(bbox=bounding_box_xyz, coords=["x", "y", "z"])
    assert len(res) == len(expected_index_xyz)
    assert (res == expected_index_xyz).all()

    # test default coords argument
    res = eptm.cut_out(bbox=bounding_box_xy)
    assert len(res) == len(expected_index_xy)
    assert (res == expected_index_xy).all()


def test_vertex_mesh():
    datasets = {}
    tri_verts = [[0, 0, 0], [1, 0, 0], [-0.5, 0.86, 1.0], [-0.5, -0.86, 1.0]]

    tri_edges = [
        [0, 1, 0, 0],
        [1, 2, 0, 0],
        [2, 0, 0, 0],
        [0, 3, 1, 0],
        [3, 1, 1, 0],
        [1, 0, 1, 0],
        [0, 2, 2, 0],
        [2, 3, 2, 0],
        [3, 0, 2, 0],
    ]

    datasets["edge"] = pd.DataFrame(
        data=np.array(tri_edges), columns=["srce", "trgt", "face", "cell"]
    )
    datasets["edge"].index.name = "edge"

    datasets["face"] = pd.DataFrame(data=np.zeros((3, 3)), columns=["x", "y", "z"])
    datasets["face"].index.name = "face"

    datasets["vert"] = pd.DataFrame(data=np.array(tri_verts), columns=["x", "y", "z"])
    datasets["vert"].index.name = "vert"

    specs = config.geometry.flat_sheet()

    eptm = Epithelium("vertex_mesh", datasets, specs, coords=["x", "y", "z"])
    SheetGeometry.update_all(eptm)

    # tested method
    res_verts, res_faces, res_normals = eptm.vertex_mesh(["x", "y", "z"])
    eptm.vertex_mesh(["x", "y", "z"], vertex_normals=False)
    res_faces = list(res_faces)

    expected_faces = [[0, 1, 2], [0, 3, 1], [0, 2, 3]]

    # floating point precision might causes issues here
    # when comparing arrays ... there seems to be
    # a built-in 1e-10 tolerance in
    # the assert_array_equal function.

    expected_normals = np.array(
        [
            [1.911111111e-01, 9.25185854e-18, 2.866666667e-01],
            [-4.16333634e-17, 0.0, 2.866666667e-01],
            [2.866666667e-01, -1.666666667e-01, 2.866666667e-01],
            [2.866666667e-01, 1.666666667e-01, 2.866666667e-01],
        ]
    )

    assert_array_equal(res_verts, np.array(tri_verts))
    assert all([res_faces[i] == expected_faces[i] for i in range(len(expected_faces))])
    assert_array_equal(
        np.round(res_normals, decimals=6), np.round(expected_normals, decimals=6)
    )


def test_ordered_edges():
    # test _ordered_edges
    # also test ordered_vert_idxs
    datasets, specs = three_faces_sheet(zaxis=True)
    eptm = Epithelium("ordered_index", datasets, specs)

    res_edges_2d = _ordered_edges(eptm.edge_df.loc[eptm.edge_df["face"] == 0])
    expected_edges_2d = [
        [0, 1, 0],
        [1, 2, 0],
        [2, 3, 0],
        [3, 4, 0],
        [4, 5, 0],
        [5, 0, 0],
    ]
    expected_vert_idxs = [idxs[0] for idxs in expected_edges_2d]
    assert res_edges_2d == expected_edges_2d
    assert expected_vert_idxs == _ordered_vert_idxs(
        eptm.edge_df.loc[eptm.edge_df["face"] == 0]
    )
    res_invalid_face = _ordered_vert_idxs(
        eptm.edge_df.loc[eptm.edge_df["face"] == 98765]
    )

    # testing the exception case in ordered_vert_idxs :
    res_invalid_face = _ordered_vert_idxs(
        eptm.edge_df.loc[eptm.edge_df["face"] == 98765]
    )
    assert np.isnan(res_invalid_face)


def test_get_prev_edges():

    tri_face = Epithelium("3", *three_faces_sheet())
    prev = get_prev_edges(tri_face).values
    expected = np.array([5, 0, 1, 2, 3, 4, 11, 6, 7, 8, 9, 10, 17, 12, 13, 14, 15, 16])
    assert_array_equal(prev, expected)


def test_get_next_edges():

    tri_face = Epithelium("3", *three_faces_sheet())
    prev = get_next_edges(tri_face).values
    expected = np.array([1, 2, 3, 4, 5, 0, 7, 8, 9, 10, 11, 6, 13, 14, 15, 16, 17, 12])
    assert_array_equal(prev, expected)


def test_get_simple_index():
    tri_face = Epithelium("3", *three_faces_sheet())
    idx = get_simple_index(tri_face.edge_df)
    assert idx.shape == (15,)

    datasets_2d, specs = three_faces_sheet(zaxis=True)
    datasets = extrude(datasets_2d, scale=1 / 3.0)
    eptm = Epithelium("3faces_3D", datasets, specs)
    idx = get_simple_index(eptm.edge_df)
    assert idx.shape == (43,)


def test_get_force_inference():
    # INIT TISSUE
    sheet = Sheet.planar_sheet_2d("jam", 10, 10, 1, 1, noise=0)
    PlanarGeometry.update_all(sheet)

    model = model_factory(
        [
            effectors.FaceAreaElasticity,
        ],
        effectors.FaceAreaElasticity,
    )

    sheet.remove(sheet.cut_out([[0, 10], [0, 10]]))
    sheet.sanitize(trim_borders=True)
    PlanarGeometry.scale(sheet, sheet.face_df.area.mean() ** -0.5, ["x", "y"])
    PlanarGeometry.center(sheet)
    PlanarGeometry.update_all(sheet)
    sheet.reset_index()
    sheet.reset_topo()
    sheet.face_df["area_elasticity"] = 1
    sheet.face_df["prefered_area"] = 1
    solver = QSSolver(with_t1=False, with_t3=False, with_collisions=False)
    solver.find_energy_min(sheet, PlanarGeometry, model, options={"gtol": 1e-8})

    sheet.vert_df.y *= 0.5
    solver.find_energy_min(sheet, PlanarGeometry, model, options={"gtol": 1e-8})
    sheet.get_force_inference(column="tension", free_border_edges=True)

    sheet = sheet.extract_bounding_box(x_boundary=[-2, 2], y_boundary=[-1, 1])

    sheet.edge_df["angle"] = (
        np.arctan2(sheet.edge_df["dx"], sheet.edge_df["dy"]) * 180 / np.pi
    )
    sheet.edge_df["angle"] = sheet.edge_df["angle"].apply(
        lambda x: 180 + x if x < 0 else x
    )
    sheet.edge_df["angle"] = sheet.edge_df["angle"].apply(
        lambda x: 180 - x if x > 90 else x
    )

    for _, edge in sheet.edge_df[sheet.edge_df.angle > 45].iterrows():
        assert edge.tension > 1.5
    for _, edge in sheet.edge_df[sheet.edge_df.angle < 45].iterrows():
        assert edge.tension < 1.5


def test_diff_srce_trgt():
    sheet = Sheet.planar_sheet_2d("jam", 10, 10, 1, 1, noise=0)
    PlanarGeometry.update_all(sheet)
    sheet.vert_df.drop(
        [
            45,
        ],
        axis=0,
        inplace=True,
    )
    assert set(sheet.edge_df.trgt) != set(sheet.vert_df.index)
    sheet.reset_index()
    assert set(sheet.edge_df.trgt) == set(sheet.vert_df.index)
