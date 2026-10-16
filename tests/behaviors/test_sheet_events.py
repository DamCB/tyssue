import os
import tempfile

import numpy as np
import pytest

from tyssue import config
from tyssue.behaviors.event_manager import EventManager, wait
from tyssue.behaviors.sheet.actions import ab_pull, decrease
from tyssue.behaviors.sheet.actions import exchange as type1_at_shorter
from tyssue.behaviors.sheet.actions import grow, increase, increase_linear_tension
from tyssue.behaviors.sheet.actions import remove as type2
from tyssue.behaviors.sheet.actions import set_value
from tyssue.behaviors.sheet.apoptosis_events import apoptosis
from tyssue.behaviors.sheet.basic_events import (
    check_tri_faces,
    contraction,
    division,
    face_elimination,
    type1_transition,
)
from tyssue.behaviors.sheet.delamination_events import constriction
from tyssue.core.sheet import Sheet
from tyssue.generation import three_faces_sheet
from tyssue.geometry.sheet_geometry import SheetGeometry as geom
from tyssue.io.hdf5 import load_datasets
from tyssue.stores import stores_dir


def test_add_events():

    manager = EventManager("face")
    initial_cell_event = [
        (division, {"face_id": 1, "geom": geom}),
        (wait, {"face_id": 3, "n_steps": 4}),
        (apoptosis, {"face_id": 5}),
        (constriction, {"face_id": 2}),
    ]
    manager.extend(initial_cell_event)
    manager.execute(None)
    manager.update()
    assert len(manager.current) == 4


def test_add_only_once():
    manager = EventManager("face")
    initial_cell_event = [
        (division, {"face_id": 1, "geom": geom}),
        (apoptosis, {"face_id": 3, "shrink_rate": 4}),
        (apoptosis, {"face_id": 3, "shrink_rate": 4}),
    ]

    manager.extend(initial_cell_event)
    manager.execute(None)
    manager.update()
    assert len(manager.current) == 2


def test_same_behavior_different_args():
    manager = EventManager("face")
    initial_cell_event = [
        (apoptosis, {"face_id": 3, "shrink_rate": 4}),
        (apoptosis, {"face_id": 3}),
    ]

    manager.extend(initial_cell_event)
    manager.execute(None)
    manager.update()
    assert len(manager.current) == 2


def test_add_only_once_current():
    manager = EventManager("face")
    manager.extend_current(
        [
            (apoptosis, {"face_id": 3, "shrink_rate": 4}),
            (apoptosis, {"face_id": 3, "shrink_rate": 4}),
            (apoptosis, {"face_id": 3, "shrink_rate": 4, "unique": False}),
        ]
    )
    # the initial wait event plus two apoptosis events
    assert len(manager.current) == 3


def test_extend_current_front():
    manager = EventManager("face")
    manager.append_current(apoptosis, face_id=1)
    manager.extend_current(
        [
            (apoptosis, {"face_id": 2}),
            (apoptosis, {"face_id": 3}),
            (apoptosis, {"face_id": 3}),  # duplicate within the batch
            (apoptosis, {"face_id": 1}),  # already queued
        ],
        front=True,
    )
    ids = [kwargs["face_id"] for _, kwargs in manager.current]
    # front events first, in their own order, ahead of wait and face 1
    assert ids == [2, 3, -1, 1]


def test_add_several_time_same_events():
    manager = EventManager("face")
    initial_cell_event = [
        (division, {"face_id": 1, "geom": geom}),
        (contraction, {"face_id": 2, "unique": False}),
        (contraction, {"face_id": 3, "unique": False}),
        (apoptosis, {"face_id": 3, "shrink_rate": 4}),
        (contraction, {"face_id": 2, "unique": False}),
    ]

    manager.extend(initial_cell_event)
    manager.execute(None)
    manager.update()
    assert len(manager.current) == 5


def test_logging():

    tf = tempfile.mktemp()
    manager = EventManager("face", tf)
    initial_cell_event = [
        (division, {"face_id": 1, "geom": geom}),
        (apoptosis, {"face_id": 3, "shrink_rate": 4}),
        (apoptosis, {"face_id": 3}),
    ]

    manager.extend(initial_cell_event)
    manager.execute(None)
    manager.update()
    with open(tf, "r") as fh:
        l0, l1, l2 = fh.readlines()

    assert l2 == "0, -1, wait\n"


def test_execute_constriction():
    h5store = os.path.join(stores_dir, "small_hexagonal.hf5")
    datasets = load_datasets(h5store, data_names=["face", "vert", "edge"])
    specs = config.geometry.cylindrical_sheet()
    sheet = Sheet("emin", datasets, specs)
    geom.update_all(sheet)
    sheet.settings["constriction"] = {
        "contractile_increase": 5,
        "critical_area": 5,
        "max_traction": 10,
    }
    sheet.face_df["id"] = sheet.face_df.index.values
    init_nb_cells = len(sheet.face_df)

    sheet.face_df["is_mesoderm"] = 0
    sheet.vert_df["radial_tension"] = 0.0
    manager = EventManager("face")
    face_id = 17
    sheet.face_df.loc[face_id, "is_mesoderm"] = 1

    sheet.settings["constriction"].update({"face_id": face_id})
    initial_cell_event = (constriction, sheet.settings["constriction"])
    manager.current.append(initial_cell_event)
    manager.execute(sheet)
    manager.update()
    assert len(manager.current) > 0

    for i in range(20):
        manager.execute(sheet)
        manager.update()
    if len(sheet.face_df) == init_nb_cells:
        assert len(manager.current) > 0


def test_execute_apoptosis():
    h5store = os.path.join(stores_dir, "small_hexagonal.hf5")
    datasets = load_datasets(h5store, data_names=["face", "vert", "edge"])
    specs = config.geometry.cylindrical_sheet()
    sheet = Sheet("emin", datasets, specs)
    geom.update_all(sheet)
    sheet.settings["apoptosis"] = {"contractile_increase": 2.0, "critical_area": 0.1}
    sheet.face_df["id"] = sheet.face_df.index.values
    init_nb_faces = len(sheet.face_df)
    manager = EventManager("face")
    face_id = 17
    face_area = sheet.face_df.loc[face_id, "area"]
    initial_nbsides = sheet.face_df.loc[face_id, "num_sides"]

    sheet.settings["apoptosis"].update({"face_id": face_id})
    initial_cell_event = (apoptosis, sheet.settings["apoptosis"])

    manager.current.append(initial_cell_event)
    manager.execute(sheet)
    manager.update()
    assert len(manager.current) > 0
    manager.execute(sheet)
    manager.update()

    sheet.settings["apoptosis"] = {
        "contractile_increase": 2.0,
        "critical_area": 2 * face_area,
    }
    manager.current.clear()
    sheet.settings["apoptosis"].update({"face_id": face_id})
    modified_cell_event = (apoptosis, sheet.settings["apoptosis"])

    manager.current.append(modified_cell_event)
    manager.execute(sheet)
    manager.update()
    next_nbsides = sheet.face_df.loc[sheet.idx_lookup(face_id, "face"), "num_sides"]
    i = 1
    while next_nbsides > 3:
        assert next_nbsides == initial_nbsides - i
        assert len(manager.current) > 0
        i = i + 1
        manager.execute(sheet)
        manager.update()
        face = sheet.idx_lookup(face_id, "face")
        if face is not None:
            next_nbsides = sheet.face_df.loc[face, "num_sides"]
        else:
            break
        if i > 20:
            raise RecursionError
    manager.execute(sheet)
    manager.update()
    assert len(sheet.face_df) < init_nb_faces


def test_execute_division():

    sheet = Sheet("emin", *three_faces_sheet())
    geom.update_all(sheet)
    sheet.face_df["vol"] = 1.0
    sheet.specs["face"]["prefered_vol"] = 1.0
    sheet.face_df["prefered_vol"] = 1.0

    sheet.face_df["id"] = sheet.face_df.index.values
    manager = EventManager("face")
    face_id = 1
    event = (division, {"face_id": face_id, "growth_rate": 1.2, "critical_vol": 1.5})
    manager.current.append(event)
    V0 = sheet.face_df.loc[1, "prefered_vol"]
    manager.execute(sheet)
    manager.update()
    assert sheet.face_df.loc[1, "prefered_vol"] == V0 * 1.2
    manager.execute(sheet)
    manager.update()
    assert sheet.face_df.loc[1, "prefered_vol"] == V0 * 1.44
    sheet.face_df.loc[1, "vol"] *= 1.6
    manager.execute(sheet)
    assert sheet.Nf == 4


def test_execute_contraction():
    sheet = Sheet("emin", *three_faces_sheet())
    geom.update_all(sheet)
    sheet.face_df["contractility"] = 1.12

    sheet.face_df["id"] = sheet.face_df.index.values
    manager = EventManager("face")
    face_id = 1
    event = (
        contraction,
        {"face_id": face_id, "contractile_increase": 0.2, "critical_area": 1e-2},
    )
    manager.current.append(event)
    manager.execute(sheet)
    manager.update()
    assert sheet.face_df.loc[face_id, "contractility"] == 1.32


def test_type1_transition():
    sheet = Sheet("emin", *three_faces_sheet())
    geom.update_all(sheet)

    sheet.face_df["id"] = sheet.face_df.index.values
    manager = EventManager("face")
    face_id = 1
    nb_neighbors_start = len(sheet.edge_df[sheet.edge_df["face"] == face_id])
    edge_to_modify = sheet.edge_df[sheet.edge_df["face"] == face_id].index[0]
    sheet.edge_df.loc[edge_to_modify, "length"] = 0.2
    event = (type1_transition, {"face_id": face_id, "critical_length": 0.3})
    manager.current.append(event)
    manager.execute(sheet)
    manager.update()

    nb_neighbors_end = len(sheet.edge_df[sheet.edge_df["face"] == face_id])

    assert nb_neighbors_end == nb_neighbors_start - 1


def test_check_tri_faces():
    h5store = os.path.join(stores_dir, "small_hexagonal.hf5")
    datasets = load_datasets(h5store, data_names=["face", "vert", "edge"])
    specs = config.geometry.cylindrical_sheet()
    sheet = Sheet("emin", datasets, specs)
    initial_nb_cells = len(sheet.face_df)
    nb_tri_cells = len(sheet.face_df[(sheet.face_df["num_sides"] < 4)])
    geom.update_all(sheet)
    sheet.face_df["id"] = sheet.face_df.index.values
    manager = EventManager("face")
    manager.current.append((check_tri_faces, {}))
    manager.execute(sheet)
    manager.update()

    manager.execute(sheet)
    manager.update()
    assert len(sheet.face_df) == initial_nb_cells - nb_tri_cells


def test_face_elimination():
    sheet = Sheet("emin", *three_faces_sheet())
    sheet.face_df["id"] = sheet.face_df.index
    face_id = 0
    manager = EventManager("face")
    event = (face_elimination, {"face_id": face_id})
    manager.current.append(event)
    manager.execute(sheet)
    manager.update()
    assert sheet.Nf == 2
    assert sheet.Nv == 8
    assert sheet.Ne == 10


def test_type1_at_shorter():

    sheet = Sheet("emin", *three_faces_sheet())
    sheet.vert_df.loc[0, "x"] += 0.5
    geom.update_all(sheet)
    type1_at_shorter(sheet, 0, geom)
    np.all(sheet.face_df.num_sides.values == [5, 7, 5])


def test_remove_face():

    sheet = Sheet("emin", *three_faces_sheet())
    type2(sheet, 0, geom)
    assert sheet.Nf == 2
    assert sheet.Nv == 8
    assert sheet.Ne == 10


def test_ab_pull():

    sheet = Sheet("emin", *three_faces_sheet())
    sheet.vert_df["radial_tension"] = 1.0
    ab_pull(sheet, 0, 1.0, distributed=False)
    np.testing.assert_array_equal(
        sheet.vert_df.loc[0:5, "radial_tension"], np.ones(6) * 2.0
    )
    ab_pull(sheet, 0, 3.0, distributed=True)
    np.testing.assert_array_equal(
        sheet.vert_df.loc[0:5, "radial_tension"], np.ones(6) * 2.5
    )


def test_set_value():
    sheet = Sheet("emin", *three_faces_sheet())
    sheet.face_df["prefered_vol"] = 1.0
    set_value(sheet, "face", 0, 12.0, "prefered_vol")
    assert sheet.face_df.loc[0, "prefered_vol"] == 12.0
    assert sheet.face_df.loc[1, "prefered_vol"] == 1.0


def test_increase():
    sheet = Sheet("emin", *three_faces_sheet())
    sheet.face_df["prefered_vol"] = 1.0
    increase(sheet, "face", 0, 1.2, "prefered_vol")
    assert sheet.face_df.loc[0, "prefered_vol"] == 1.2


def test_decrease():
    sheet = Sheet("emin", *three_faces_sheet())
    sheet.face_df["prefered_vol"] = 1.0
    decrease(sheet, "face", 0, 1.6, "prefered_vol")
    assert sheet.face_df.loc[0, "prefered_vol"] == 0.625


def test_increase_line_tension():
    sheet = Sheet("emin", *three_faces_sheet())

    sheet.edge_df["line_tension"] = 1
    edges = sheet.edge_df[sheet.edge_df["face"] == 4]
    for index, edge in edges.iterrows():
        sheet.edge_df.loc[edge, "line_tension"] == 2

    increase_linear_tension(sheet, 0, 4)

    edges = sheet.edge_df[sheet.edge_df["face"] == 4]
    for index, edge in edges.iterrows():
        angle_ = np.arctan2(sheet.edge_df.dx, sheet.edge_df.dy)
        if np.abs(angle_) < np.pi / 4:
            assert sheet.edge_df.loc[edge, "line_tension"] == 8


def test_grow():
    sheet = Sheet("emin", *three_faces_sheet())
    sheet.face_df["prefered_vol"] = 1.0
    with pytest.warns(UserWarning):
        grow(sheet, 0, 1.2)
        assert sheet.face_df.loc[0, "prefered_vol"] == 1.2


def _square_triangle(gap=0.005):
    """A square and a triangle whose tip is `gap` away from the square's right
    (free) edge."""
    import pandas as pd

    from tyssue.config.geometry import planar_spec
    from tyssue.geometry.planar_geometry import PlanarGeometry

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


def _two_patches(gap):
    """Two hexagonal patches side by side, `gap` apart along x."""
    import pandas as pd

    from tyssue.config.geometry import planar_spec
    from tyssue.geometry.planar_geometry import PlanarGeometry

    patch = Sheet.planar_sheet_2d("patch", 5, 5, 1, 1)
    patch.sanitize(trim_borders=True, order_edges=True)
    width = patch.vert_df["x"].max() - patch.vert_df["x"].min()
    datasets = {}
    for name in ("vert", "edge", "face"):
        left = patch.datasets[name].copy()
        right = patch.datasets[name].copy()
        right.index = right.index + len(left)
        if name in ("vert", "face"):
            right["x"] += width + gap
        if name == "edge":
            right[["srce", "trgt"]] += patch.Nv
            right["face"] += patch.Nf
        datasets[name] = pd.concat([left, right])
    sheet = Sheet("two", datasets, specs=planar_spec(), coords=["x", "y"])
    sheet.reset_index()
    sheet.reset_topo()
    PlanarGeometry.update_all(sheet)
    return sheet


def _brute_force_t3(sheet, d_min):
    """Every (vertex, edge) pair eligible for a T3, by exhaustive search."""
    edge_df = sheet.edge_df
    pos = sheet.vert_df[sheet.coords]
    free = edge_df[edge_df["opposite"] == -1]
    border = set(free["srce"]) | set(free["trgt"])
    linked = set(zip(edge_df["srce"], edge_df["trgt"]))
    face_verts = edge_df.groupby("face")["srce"].apply(set)
    pairs = set()
    for edge, (a, b, face) in free[["srce", "trgt", "face"]].iterrows():
        pa, pb = pos.loc[a].to_numpy(), pos.loc[b].to_numpy()
        for v in border:
            if v in (a, b) or v in face_verts[face]:
                continue
            # a -> v or v -> b would be created twice
            if {(a, v), (v, b)} & linked:
                continue
            pv = pos.loc[v].to_numpy()
            t = np.dot(pv - pa, pb - pa) / np.dot(pb - pa, pb - pa)
            if np.linalg.norm(pa + t * (pb - pa) - pv) < d_min and 0.05 < t < 0.95:
                pairs.add((v, edge))
    return pairs


def test_t3_candidates_match_brute_force(monkeypatch):
    from tyssue.behaviors.sheet import actions
    from tyssue.geometry.planar_geometry import PlanarGeometry

    builds = []
    neighbor_list = actions._t3_neighbor_list

    def counting(*args, **kwargs):
        builds.append(1)
        return neighbor_list(*args, **kwargs)

    monkeypatch.setattr(actions, "_t3_neighbor_list", counting)

    rng = np.random.default_rng(42)
    sheet = _two_patches(gap=0.4)
    d_min, skin = 0.3, 0.1
    right = sheet.vert_df["x"] > sheet.vert_df["x"].mean()
    n_steps, n_found = 300, 0
    for step in range(n_steps):
        noise = rng.normal(0, 0.004, (sheet.Nv, 2))
        if step % 50 == 25:
            noise *= 20  # a jump larger than skin / 2, forces a rebuild
        sheet.vert_df[["x", "y"]] += noise
        # the patches drift towards and away from each other
        sheet.vert_df.loc[right, "x"] -= 0.002 * np.sign(np.sin(step / 40))
        PlanarGeometry.update_all(sheet)

        verts, edges, dists = actions.find_t3_candidates(sheet, d_min, skin)
        got = set(
            zip(sheet.vert_df.index[verts].tolist(), sheet.edge_df.index[edges].tolist())
        )
        assert got == _brute_force_t3(sheet, d_min), f"step {step}"
        assert np.all(np.diff(dists) >= 0)
        n_found += len(got)
    assert n_found > 0
    # the candidate list is re-used most of the time
    assert 1 < len(builds) < n_steps / 3


def test_t3_transitions():
    from tyssue.behaviors.sheet.actions import t3_transitions
    from tyssue.geometry.planar_geometry import PlanarGeometry

    sheet = _square_triangle(gap=0.005)
    sheet.settings["t3_threshold"] = 0.01
    ne = sheet.Ne
    assert t3_transitions(sheet) == 1
    PlanarGeometry.update_all(sheet)
    assert sheet.Ne == ne + 1
    assert sheet.validate()
    assert t3_transitions(sheet) == 0

    far = _square_triangle(gap=0.05)
    far.settings["t3_threshold"] = 0.01
    assert t3_transitions(far) == 0


def test_t3_transitions_zip_linked_free_edges():
    """A vertex linked to the end of a nearby free edge by a free edge of its
    own face is merged too: the two free edges become one shared edge."""
    import pandas as pd

    from tyssue.behaviors.sheet.actions import t3_transitions
    from tyssue.config.geometry import planar_spec
    from tyssue.geometry.planar_geometry import PlanarGeometry

    # a square and a triangle sharing the corner 2; the triangle's 2 -> 4 runs
    # down along the square's 1 -> 2, 4 is 0.005 away from it
    vert = pd.DataFrame({"x": [0.0, 1, 1, 0, 1.005, 1.8], "y": [0.0, 0, 1, 1, 0.4, 0.5]})
    edge = pd.DataFrame(
        {"srce": [0, 1, 2, 3, 2, 4, 5], "trgt": [1, 2, 3, 0, 4, 5, 2],
         "face": [0, 0, 0, 0, 1, 1, 1]}
    )
    face = pd.DataFrame({"x": [0.5, 1.3], "y": [0.5, 0.6]})
    for df, name in ((vert, "vert"), (edge, "edge"), (face, "face")):
        df.index.name = name
    sheet = Sheet("sq_wedge", {"vert": vert, "edge": edge, "face": face},
                  specs=planar_spec(), coords=["x", "y"])
    PlanarGeometry.update_all(sheet)
    sheet.settings["t3_threshold"] = 0.01

    assert t3_transitions(sheet) == 1
    PlanarGeometry.update_all(sheet)
    assert sheet.validate()
    assert (sheet.edge_df["opposite"] >= 0).sum() == 2
    assert t3_transitions(sheet) == 0


def test_t3_transitions_closed_sheet():
    from tyssue.behaviors.sheet.actions import t3_transitions
    from tyssue.generation import ellipsoid_sheet

    sheet = ellipsoid_sheet(3, 3, 4, 12)
    assert t3_transitions(sheet) == -1
    assert getattr(sheet, "_t3_cache", None) is None


def test_t3_transitions_sheet_only():
    from tyssue.behaviors.sheet.actions import t3_transitions
    from tyssue.core.monolayer import Monolayer
    from tyssue.generation import extrude

    sheet = _square_triangle()
    sheet.vert_df["z"] = 0.0
    sheet.face_df["z"] = 0.0
    sheet.coords = ["x", "y", "z"]
    mono = Monolayer("mono", extrude(sheet.datasets, method="translation"))
    with pytest.raises(TypeError):
        t3_transitions(mono)


def test_t3_event_in_euler_solver():
    from tyssue.collisions.intersections import find_intersections
    from tyssue.dynamics import effectors, model_factory
    from tyssue.geometry.planar_geometry import PlanarGeometry
    from tyssue.solvers.viscous import EulerSolver

    sheet = _square_triangle(gap=0.02)
    model = model_factory([effectors.LineTension, effectors.FaceAreaElasticity])
    sheet.update_specs(model.specs, reset=True)
    sheet.face_df["prefered_area"] = sheet.face_df["area"]
    sheet.edge_df["line_tension"] = 0.0
    sheet.vert_df["viscosity"] = 1.0
    sheet.settings["t3_threshold"] = 0.01
    PlanarGeometry.update_all(sheet)

    solver = EulerSolver(sheet, PlanarGeometry, model, with_t3=True)
    assert solver.manager is not None
    ne = sheet.Ne
    for _ in range(5):
        # the triangle moves towards the square
        sheet.vert_df.loc[[4, 5, 6], "x"] -= 0.004
        solver.solve(tf=solver.prev_t + 0.01, dt=0.01)
    assert sheet.Ne == ne + 1
    assert sheet.validate()
    assert find_intersections(sheet).clean


def test_detach_boundary_setting():
    from tyssue.behaviors.sheet.actions import detach_vertices, t3_transitions
    from tyssue.geometry.planar_geometry import PlanarGeometry

    sheet = _two_patches(gap=0.004)
    sheet.settings.update(t3_threshold=0.01, p_4=1e9, p_5p=1e9, dt=1.0)
    assert t3_transitions(sheet) > 0
    PlanarGeometry.update_all(sheet)
    nv = sheet.Nv

    kept = sheet.copy()
    kept.settings.update(sheet.settings, detach_boundary=False)
    PlanarGeometry.update_all(kept)
    detach_vertices(kept)
    assert kept.Nv == nv

    detach_vertices(sheet)  # default: the T3 junction is a rank 4 vertex
    assert sheet.Nv > nv


def test_euler_solver_with_t3_sheet_only():
    from tyssue.core.monolayer import Monolayer
    from tyssue.generation import extrude
    from tyssue.geometry.bulk_geometry import MonolayerGeometry
    from tyssue.solvers.viscous import EulerSolver

    sheet = _square_triangle()
    sheet.vert_df["z"] = 0.0
    sheet.face_df["z"] = 0.0
    mono = Monolayer("mono", extrude(sheet.datasets, method="translation"))
    with pytest.raises(ValueError, match="Sheet"):
        EulerSolver(mono, MonolayerGeometry, None, with_t3=True)
