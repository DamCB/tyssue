"""Tests for the monolayer tissue-closure / fusion behaviors.

Lateral-face fusion (:mod:`tyssue.topology.bulk_topology`), the ``fuse``
event, and the lift of a monolayer by apical constriction.

Some tests render a 3D gif to ``tests/behaviors/output/`` when imagemagick is
installed.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.spatial import Voronoi

from tyssue import Monolayer, MonolayerGeometry, Sheet, config
from tyssue.behaviors.event_manager import EventManager
from tyssue.behaviors.monolayer.closure_events import fuse
from tyssue.core.history import History
from tyssue.dynamics import effectors
from tyssue.dynamics.factory import model_factory
from tyssue.generation import extrude, from_2d_voronoi, hexa_disk
from tyssue.solvers.quasistatic import QSSolver
from tyssue.topology.bulk_topology import (
    _cell_adjacency,
    _free_lateral_faces,
    _split_lateral_face,
    all_lateral_fusions,
    find_fusion_nucleations,
    find_fusion_propagations,
    find_fusion_splits,
    fuse_lateral_faces,
)

geom = MonolayerGeometry

OUTPUT_DIR = Path(__file__).parent / "output"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _flat_monolayer(nx=7, ny=5):
    """A flat rectangular monolayer extruded from a planar sheet."""
    sheet = Sheet.planar_sheet_3d("flat", nx, ny, 1, 1)
    sheet.sanitize()
    mono = Monolayer("mono", extrude(sheet.datasets, method="translation"),
                     config.geometry.bulk_spec())
    geom.center(mono)
    geom.update_all(mono)
    mono.settings["threshold_length"] = 1e-2
    return mono


def _n_cell_components(mono):
    """Number of connected components of the cell-cell adjacency graph."""
    import networkx as nx

    mono.get_opposite_faces()
    graph = nx.Graph()
    graph.add_nodes_from(mono.cell_df.index)
    edge2cell = mono.edge_df.groupby("face")["cell"].first()
    for face, opp in mono.face_df["opposite"].items():
        if opp != -1:
            graph.add_edge(edge2cell[face], edge2cell[opp])
    return nx.number_connected_components(graph)


def _n_lateral_border(mono):
    mono.get_opposite_faces()
    border = mono.face_df["opposite"] == -1
    return int((border & (mono.face_df["segment"] == "lateral")).sum())


def _n_vertex_components(mono):
    """Number of connected components when cells are joined by a shared vertex."""
    import networkx as nx

    graph = nx.Graph()
    graph.add_nodes_from(mono.cell_df.index)
    for cells in mono.edge_df.groupby("srce")["cell"].apply(set):
        cells = list(cells)
        for i in range(len(cells)):
            for j in range(i + 1, len(cells)):
                graph.add_edge(cells[i], cells[j])
    return nx.number_connected_components(graph)


def _incise(mono, axis="x"):
    """Splits a monolayer into two halves along the median of `axis`.

    Shared vertices are duplicated, leaving two coincident free walls.
    Returns the cells on each side.
    """
    mid = mono.cell_df[axis].median()
    left = set(mono.cell_df[mono.cell_df[axis] < mid].index)
    right = set(mono.cell_df[mono.cell_df[axis] >= mid].index)

    vert_cells = mono.edge_df.groupby("srce")["cell"].apply(set)
    shared = [v for v, cells in vert_cells.items() if (cells & left) and (cells & right)]

    new_index = mono.vert_df.index.max() + 1
    right_mask = mono.edge_df["cell"].isin(right)
    for vert in shared:
        new_row = mono.vert_df.loc[[vert]].copy()
        new_row.index = [new_index]
        mono.vert_df = pd.concat([mono.vert_df, new_row])
        mono.edge_df.loc[right_mask & (mono.edge_df["srce"] == vert), "srce"] = new_index
        mono.edge_df.loc[right_mask & (mono.edge_df["trgt"] == vert), "trgt"] = new_index
        new_index += 1

    mono.reset_index()
    mono.reset_topo()
    mono.get_opposite_faces()
    geom.update_all(mono)
    # recompute the (reindexed) cell sets
    left = set(mono.cell_df[mono.cell_df[axis] < mid].index)
    right = set(mono.cell_df[mono.cell_df[axis] >= mid].index)
    return left, right


def _half_verts(mono, cells):
    return mono.edge_df[mono.edge_df["cell"].isin(cells)]["srce"].unique()


def _render(history, name):
    """Renders a 3D gif of `history`; skips if imagemagick is unavailable."""
    from tyssue.draw.plt_draw import create_gif_3d, imagemagick_cmd

    if imagemagick_cmd() is None:
        pytest.skip("imagemagick not available, skipping video rendering")

    OUTPUT_DIR.mkdir(exist_ok=True)
    output = OUTPUT_DIR / name
    history.update_datasets()
    # wireframe: faces can be degenerate while the topology changes
    create_gif_3d(
        history,
        output.as_posix(),
        num_frames=len(history.time_stamps),
        coords=["x", "y", "z"],
        draw_order=("edge", "vert"),
        edge={"visible": True, "color": "#1f3b5c", "width": 1.0},
        vert={"visible": True, "color": "#c1121f", "s": 8},
        view_angle=(25, -60),
        dpi=80,
    )
    assert output.exists()


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
def _offset_fronts(nx=6, ny=5, gap=0.0, shear=0.5):
    """A monolayer cut in two, with one half sheared so the walls are not
    coincident. Returns (mono, left_cells, right_cells)."""
    mono = _flat_monolayer(nx, ny)
    left, right = _incise(mono)
    left_v = _half_verts(mono, left)
    right_v = _half_verts(mono, right)
    # shear the right half across the cut, then overlap the walls
    mono.vert_df.loc[right_v, "y"] += shear
    mono.vert_df.loc[left_v, "x"] -= gap / 2
    mono.vert_df.loc[right_v, "x"] += gap / 2
    geom.update_all(mono)
    return mono, left, right


def _pocket_monolayer():
    """Cell C's wide lateral wall apposed to the walls of cells A and B, with
    both end corners shared. Cells are A=0, B=1, C=2.
    """
    verts = {0: (0, 0), 1: (1, 0), 2: (1, 1), 3: (0, 1),
             4: (1, 2), 5: (0, 2), 6: (2, 0), 7: (2, 2)}
    faces = {0: [0, 1, 2, 3], 1: [3, 2, 4, 5], 2: [1, 6, 7, 4]}
    vert_df = pd.DataFrame(
        [(x, y, 0.0) for x, y in verts.values()], columns=list("xyz"), index=list(verts)
    )
    rows = [
        (s, t, f) for f, loop in faces.items() for s, t in zip(loop, loop[1:] + loop[:1])
    ]
    edge_df = pd.DataFrame(rows, columns=["srce", "trgt", "face"])
    face_df = pd.DataFrame(index=list(faces))
    face_df.index.name = "face"
    sheet = Sheet("pocket", {"vert": vert_df, "edge": edge_df, "face": face_df})
    sheet.reset_index()
    sheet.reset_topo()
    mono = Monolayer(
        "pocket",
        extrude(sheet.datasets, method="translation", vector=[0, 0, -1]),
        config.geometry.bulk_spec(),
    )
    geom.update_all(mono)
    mono.settings["threshold_length"] = 1e-2
    return mono


def test_split_lateral_face_divides_a_wall():
    """Splitting a free lateral wall yields two free lateral quads of the same
    cell (two new vertices, one new face) and keeps the mesh valid."""
    mono = _flat_monolayer()
    mono.get_opposite_faces()
    face = _free_lateral_faces(mono)[0]
    cell = int(mono.edge_df.loc[mono.edge_df["face"] == face, "cell"].iloc[0])
    nv, nf, nc = mono.Nv, mono.Nf, mono.Nc

    na, nb, new_face = _split_lateral_face(mono, face)
    geom.update_all(mono)

    assert mono.validate()
    assert mono.Nv == nv + 2  # one apical + one basal vertex
    assert mono.Nf == nf + 1  # the wall became two faces
    assert mono.Nc == nc      # no cell created or lost
    # both halves are free lateral quads of the original cell
    for f in (face, new_face):
        edges = mono.edge_df[mono.edge_df["face"] == f]
        assert mono.face_df.loc[f, "segment"] == "lateral"
        assert mono.face_df.loc[f, "opposite"] == -1
        assert int(edges["cell"].iloc[0]) == cell
        assert edges.shape[0] == 4


def test_find_fusion_splits_flags_the_wide_wall():
    """In a 2-vs-1 pocket only the single wide wall (apposed to two non-adjacent
    cells) is flagged for splitting."""
    mono = _pocket_monolayer()
    splits = find_fusion_splits(mono, theta_fold=20.0)
    assert len(splits) == 1
    # the flagged face is cell C's (cell 2) wall
    wide = splits[0]
    assert int(mono.edge_df.loc[mono.edge_df["face"] == wide, "cell"].iloc[0]) == 2


def test_split_closes_a_2_vs_1_pocket():
    """A wide wall apposed to two narrower walls cannot fuse directly, but
    splitting it lets the pocket zip shut, joining the cell to both partners."""
    mono = _pocket_monolayer()
    mono.get_opposite_faces()
    # before: C (cell 2) is adjacent to neither A (0) nor B (1)
    assert _cell_adjacency(mono) == {frozenset((0, 1))}
    # and the apposed walls cannot be fused without splitting first
    stuck = _pocket_monolayer()
    for g, f in find_fusion_propagations(stuck, theta_fold=20.0):
        assert fuse_lateral_faces(stuck, g, f) == -1

    n = all_lateral_fusions(mono, d_max=0.5, theta_fold=20.0)
    geom.update_all(mono)

    assert n >= 1
    assert mono.validate()
    assert mono.Nc == 3
    # the wide cell is now joined to both of its partners across the closed seam
    adjacency = _cell_adjacency(mono)
    assert frozenset((0, 2)) in adjacency
    assert frozenset((1, 2)) in adjacency


def test_fuse_lateral_faces_joins_two_cells():
    """A single lateral-face weld makes two apposed cells share an interface and
    keeps the mesh valid."""
    mono, left, right = _offset_fronts(gap=-0.2)  # walls overlapping a little
    nucleations = find_fusion_nucleations(mono, d_max=0.5)
    assert len(nucleations) >= 1
    g, f = nucleations[0]
    # the two faces belong to different, not-yet-adjacent cells
    cell_g = mono.edge_df.loc[mono.edge_df["face"] == g, "cell"].iloc[0]
    cell_f = mono.edge_df.loc[mono.edge_df["face"] == f, "cell"].iloc[0]
    assert cell_g != cell_f
    nv_before = mono.Nv

    assert fuse_lateral_faces(mono, g, f) == 0
    geom.update_all(mono)

    assert mono.validate()
    # welding removes vertices (no new edges/verts are created)
    assert mono.Nv < nv_before


def test_find_fusion_nucleations_on_offset_fronts():
    """Apposed lateral faces are detected across non-coincident (sheared) walls,
    only between different, non-adjacent cells."""
    mono, left, right = _offset_fronts(gap=-0.4)  # walls overlapping
    nucleations = find_fusion_nucleations(mono, d_max=0.5)
    assert len(nucleations) >= 1
    for g, f in nucleations:
        cell_g = mono.edge_df.loc[mono.edge_df["face"] == g, "cell"].iloc[0]
        cell_f = mono.edge_df.loc[mono.edge_df["face"] == f, "cell"].iloc[0]
        assert cell_g != cell_f
        # the two faces are both free lateral walls
        assert mono.face_df.loc[g, "segment"] == "lateral"
        assert mono.face_df.loc[f, "segment"] == "lateral"


def test_propagation_zips_a_registered_seam():
    """Once a seam nucleates, the fold-angle (propagation) trigger zips the rest
    of a registered seam, and same-front neighbours never fuse."""
    mono, left, right = _offset_fronts(gap=0.0, shear=0.0)  # coincident walls
    assert _n_vertex_components(mono) == 2

    n_nuc = n_prop = 0
    for _ in range(60):
        props = find_fusion_propagations(mono, theta_fold=20.0)
        nucs = find_fusion_nucleations(mono, d_max=0.4)
        done = False
        for g, f in props:
            if fuse_lateral_faces(mono, g, f) == 0:
                n_prop += 1
                done = True
                break
        if not done:
            for g, f in nucs:
                if fuse_lateral_faces(mono, g, f) == 0:
                    n_nuc += 1
                    done = True
                    break
        geom.update_all(mono)
        if not done:
            break

    # the seam nucleated and then propagated along shared edges
    assert n_nuc >= 1
    assert n_prop >= 1
    assert _n_vertex_components(mono) == 1
    assert mono.validate()


def test_offset_fronts_fuse_via_event_manager():
    """Two sheared fronts are joined into one tissue by lateral-face fusion.

    The walls are offset by half a cell. Fusion runs through the ``fuse``
    event.
    """
    mono, left, right = _offset_fronts(gap=0.8)
    n_cells = mono.Nc
    assert _n_vertex_components(mono) == 2  # the incision separated the fronts

    left_v = _half_verts(mono, left)
    right_v = _half_verts(mono, right)

    manager = EventManager()
    manager.append(fuse, d_max=0.4)

    history = History(mono)
    history.record(time_stamp=0.0)

    gap, n_steps, step, moved = 0.8, 14, 0.15, 0.0
    for t in range(1, n_steps + 1):
        if moved < gap + 0.2:  # keep advancing slightly past contact
            mono.vert_df.loc[left_v, "x"] += step / 2
            mono.vert_df.loc[right_v, "x"] -= step / 2
            moved += step
            geom.update_all(mono)

        manager.update()
        manager.execute(mono)
        geom.update_all(mono)
        left = set(c for c in left if c in mono.cell_df.index)
        right = set(c for c in right if c in mono.cell_df.index)
        left_v = _half_verts(mono, left)
        right_v = _half_verts(mono, right)
        history.record(time_stamp=float(t))

    # the fronts are joined into a single tissue, no cells lost, mesh valid
    assert _n_vertex_components(mono) == 1
    assert mono.Nc == n_cells
    assert mono.validate()

    _render(history, "offset_fronts_fusion.gif")


def _circular_monolayer(num_t=40, radius=6.75):
    """A large flat circular monolayer built from a hexagonal disk.

    With the defaults, ~118 cells of apical area ~1 and unit thickness.
    """
    points = hexa_disk(num_t, radius)
    sheet = Sheet("disk", from_2d_voronoi(Voronoi(points)))
    sheet.sanitize(trim_borders=True, order_edges=True)
    sheet.reset_index()
    sheet.reset_topo()
    sheet.vert_df["z"] = 0.0

    mono = Monolayer(
        "disk",
        extrude(sheet.datasets, method="translation", vector=[0, 0, -1.0]),
        config.geometry.bulk_spec(),
    )
    geom.update_all(mono)
    mono.settings["threshold_length"] = 0.05
    return mono


def _apical_constriction_model(mono):
    """Builds the apical-constriction energy model and its parameters.

    Area elasticity, volume elasticity, line tension and contractility, with
    the prefered areas and volumes set to the current values.
    """
    model = model_factory(
        [
            effectors.LineTension,
            effectors.FaceContractility,
            effectors.FaceAreaElasticity,
            effectors.CellVolumeElasticity,
        ],
        effectors.CellVolumeElasticity,
    )
    mono.update_specs(model.specs, reset=False)

    # minimal relaxation: prefered area/volume == current
    mono.face_df["area_elasticity"] = 1.0
    mono.face_df["prefered_area"] = mono.face_df["area"]
    mono.cell_df["vol_elasticity"] = 1.0
    mono.cell_df["prefered_vol"] = mono.cell_df["vol"]
    mono.vert_df["viscosity"] = 1.0

    # apical constriction: line tension = 1 on apical edges, 0 everywhere else
    mono.face_df["contractility"] = 0.0
    mono.edge_df["line_tension"] = 0.0
    mono.edge_df.loc[mono.apical_edges, "line_tension"] = 1.0

    geom.update_all(mono)
    return model


def test_circular_tissue_lifts_by_apical_constriction():
    """A large circular monolayer lifts off the plane by apical constriction.

    The apical line tension is ramped up with :class:`QSSolver`. The tissue
    forms an open cup, keeping its cells and a valid mesh; no fusion happens.
    """
    mono = _circular_monolayer()
    n_cells = mono.Nc
    assert n_cells >= 100
    # cells were generated with mean apical area ~1 and unit thickness
    assert abs(mono.face_df.loc[mono.apical_faces, "area"].mean() - 1.0) < 0.1
    assert abs(mono.cell_df["vol"].mean() - 1.0) < 0.15

    # a small initial dome breaks the up/down symmetry of the flat disk
    r = np.hypot(mono.vert_df["x"], mono.vert_df["y"])
    mono.vert_df["z"] += 0.5 * (1.0 - (r / r.max()) ** 2)
    geom.update_all(mono)
    z_span_seed = float(np.ptp(mono.vert_df["z"].to_numpy()))

    model = _apical_constriction_model(mono)

    # ramp the apical line tension, relaxing to the energy minimum each step
    solver = QSSolver()
    history = History(mono)
    history.record(time_stamp=0.0)
    tensions = np.linspace(0.1, 1.0, 16)
    for i, tension in enumerate(tensions, start=1):
        mono.edge_df["line_tension"] = 0.0
        mono.edge_df.loc[mono.apical_edges, "line_tension"] = tension
        geom.update_all(mono)
        energy_before = model.compute_energy(mono)
        res = solver.find_energy_min(mono, geom, model)
        # res["success"] depends on the platform; the energy must not increase
        assert np.isfinite(res["fun"])
        assert res["fun"] <= energy_before + 1e-8
        history.record(time_stamp=float(i))

    geom.update_all(mono)

    # apical line tension reached the requested value of 1 ...
    assert np.isclose(mono.edge_df.loc[mono.apical_edges, "line_tension"].mean(), 1.0)
    # ... the tissue lifted off the plane while staying intact and valid
    assert mono.Nc == n_cells
    assert mono.validate()
    z_span_final = float(np.ptp(mono.vert_df["z"].to_numpy()))
    assert z_span_final > 2.0                 # a clear lift off the plane
    assert z_span_final > z_span_seed + 0.5   # apical constriction amplified it

    _render(history, "circular_lift.gif")
