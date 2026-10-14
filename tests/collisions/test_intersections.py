"""Tests for :mod:`tyssue.collisions.intersections`.

Results are checked against slow reference implementations where they exist.
"""

import numpy as np
import pandas as pd
import pytest

from tyssue import Sheet
from tyssue.behaviors import EventManager
from tyssue.behaviors.sheet import IntersectionError, check_intersections
from tyssue.collisions import intersections as ix


# ---------------------------------------------------------------------------
def cross2(a, b):
    """z component of the 2D cross product (np.cross dropped 2-vectors)."""
    return a[0] * b[1] - a[1] * b[0]


def make_sheet(verts, faces):
    """Build a Sheet from vertex coordinates and a list of vertex rings."""
    verts = np.asarray(verts, dtype=float)
    if verts.shape[1] == 2:
        verts = np.column_stack([verts, np.zeros(len(verts))])
    vert = pd.DataFrame(verts, columns=["x", "y", "z"])

    rows = []
    for fid, ring in enumerate(faces):
        for i, v in enumerate(ring):
            rows.append((int(v), int(ring[(i + 1) % len(ring)]), fid))
    # shuffled: edge_df row order is not ring order
    rng = np.random.default_rng(0)
    rows = [rows[i] for i in rng.permutation(len(rows))]
    edge = pd.DataFrame(rows, columns=["srce", "trgt", "face"])
    face = pd.DataFrame({c: np.zeros(len(faces)) for c in ("x", "y", "z")})
    return Sheet("test", {"vert": vert, "edge": edge, "face": face})


def hexagon(cx=0.0, cy=0.0, r=1.0, n=6, phase=0.0):
    a = np.arange(n) * 2 * np.pi / n + phase
    return np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a)])


# ---------------------------------------------------------------------------
# Test A -- self-intersection
# ---------------------------------------------------------------------------
def test_regular_hexagon_is_clean():
    r = make_sheet(hexagon(), [[0, 1, 2, 3, 4, 5]]).find_intersections()
    assert r.clean
    assert len(r.folded) == 0


def test_bowtie_quad_is_flagged():
    # swapping two vertices of a square makes a bow-tie, with a zero Newell vector
    s = make_sheet([[0, 0], [1, 0], [0, 1], [1, 1]], [[0, 1, 2, 3]])
    assert list(s.find_intersections().folded) == [0]


def test_reflex_but_simple_polygon_is_not_flagged():
    """Non-convex but simple: a non-convexity screen calls this folded, a real
    self-intersection test must not."""
    poly = [[0, 0], [2, 0], [2, 2], [1, 0.5], [0, 2]]
    assert len(make_sheet(poly, [[0, 1, 2, 3, 4]]).find_intersections().folded) == 0

    P = np.array(poly)
    c = P.mean(0)
    fan = sum(abs(cross2(P[i] - c, P[(i + 1) % 5] - c)) / 2 for i in range(5))
    vec = abs(sum(cross2(P[i] - c, P[(i + 1) % 5] - c) for i in range(5)) / 2)
    assert vec / fan < 0.95, "should fool a |A_vec|/A_fan screen"


def test_open_ring_is_reported_not_silently_tested():
    s = make_sheet(hexagon(), [[0, 1, 2, 3, 4, 5]])
    s.edge_df.loc[0, "trgt"] = s.edge_df.loc[0, "srce"]
    r = s.find_intersections()
    assert len(r.open_rings) == 1
    assert len(r.folded) == 0


def test_validate_passes_a_bowtie_but_find_intersections_does_not():
    """The two checks are complementary: ``validate`` is combinatorial."""
    s = make_sheet([[0, 0], [1, 0], [0, 1], [1, 1]], [[0, 1, 2, 3]])
    assert s.validate(), "a bow-tie is a closed polygon combinatorially"
    assert len(s.find_intersections().folded) == 1


# ---------------------------------------------------------------------------
# Test B -- containment
# ---------------------------------------------------------------------------
def _hex_with_probe(px, py):
    verts = np.vstack([hexagon(), [[px, py]]])
    return make_sheet(verts, [[0, 1, 2, 3, 4, 5], [6, 0, 1]])


def test_vertex_just_inside_is_detected():
    r = _hex_with_probe(0.05, 0.0).find_intersections()
    assert any(f == 0 and v == 6 for f, v, _ in r.contained)


def test_vertex_just_outside_is_not_detected():
    r = _hex_with_probe(2.5, 0.0).find_intersections()
    assert not any(f == 0 and v == 6 for f, v, _ in r.contained)


def test_edge_sharing_neighbours_are_clean():
    """Shared vertices sit exactly on the neighbour's boundary. They are excluded
    by ring membership, so no tolerance is involved and the answer is exact."""
    A = hexagon(0.0, 0.0)
    cx, cy = A[0] + A[1]
    B = hexagon(cx, cy)
    assert np.allclose(B[3], A[1]) and np.allclose(B[4], A[0])
    s = make_sheet(np.vstack([A, B[[0, 1, 2, 5]]]),
                   [[0, 1, 2, 3, 4, 5], [6, 7, 8, 1, 0, 9]])
    r = s.find_intersections()
    assert len(r.contained) == 0
    assert len(r.folded) == 0


def test_winding_rule_is_nonzero_not_even_odd():
    """A doubly-wound lobe has winding 2: nonzero calls it inside (material
    covers the point), even-odd calls it outside. The choice is deliberate."""
    tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
    poly = np.vstack([tri, tri])
    assert ix._winding_inside(poly, np.array([[0.5, 0.35]]))[0]


# ---------------------------------------------------------------------------
# Test C -- overlap
# ---------------------------------------------------------------------------
def test_lens_overlap_needs_strict_mode():
    """Two thin quads crossed in an X overlap with no vertex of either inside the
    other: Test B cannot see it, Test C must."""
    a = [[-2, -0.15], [2, -0.15], [2, 0.15], [-2, 0.15]]
    b = [[-0.15, -2], [0.15, -2], [0.15, 2], [-0.15, 2]]
    s = make_sheet(a + b, [[0, 1, 2, 3], [4, 5, 6, 7]])
    assert len(s.find_intersections().contained) == 0
    assert (0, 1) in s.find_intersections(strict=True).overlaps


# ---------------------------------------------------------------------------
# Invariance and curvature
# ---------------------------------------------------------------------------
def _rot(P, axis, ang):
    axis = np.asarray(axis, float)
    axis /= np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return P @ (np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)).T


@pytest.mark.parametrize("ang", [0.3, 1.1, 2.7, 4.9])
def test_rotation_and_translation_invariance(ang):
    verts = np.column_stack([*hexagon().T, np.zeros(6)])
    verts = np.vstack([verts, [0.05, 0.0, 0.0]])
    faces = [[0, 1, 2, 3, 4, 5], [6, 0, 1]]
    base = make_sheet(verts, faces).find_intersections()
    moved = make_sheet(_rot(verts, [0.3, -0.7, 0.6], ang) + [13.0, -5.0, 2.0],
                       faces).find_intersections()
    assert list(base.folded) == list(moved.folded)
    assert sorted((f, v) for f, v, _ in base.contained) == \
           sorted((f, v) for f, v, _ in moved.contained)


def _tube_patch(th0, th1, z0, z1, R=2.5):
    th = np.array([th0, th1, th1, th0])
    z = np.array([z0, z0, z1, z1])
    return np.column_stack([R * np.cos(th), R * np.sin(th), z])


def test_periodic_seam_needs_no_special_handling():
    """A face straddling theta = +-pi behaves exactly like one that does not: the
    frame is built from local differences, so no global angle is ever taken."""
    n = 6
    th = np.linspace(-0.25, 0.25, n) + np.pi
    z = np.array([0, 1, 2, 2, 1, 0], float)
    R = 2.5
    on_seam = np.column_stack([R * np.cos(th), R * np.sin(th), z])
    off_seam = np.column_stack([R * np.cos(th - np.pi), R * np.sin(th - np.pi), z])
    ring = [list(range(n))]
    assert list(make_sheet(on_seam, ring).find_intersections().folded) == \
           list(make_sheet(off_seam, ring).find_intersections().folded)

    folded = on_seam.copy()
    folded[[2, 3]] = folded[[3, 2]]
    assert len(make_sheet(folded, ring).find_intersections().folded) == 1


def test_wide_face_on_a_tube_is_not_a_false_positive():
    """A face spanning a wide arc is reported as undecidable, not folded."""
    wide = _tube_patch(0.0, 1.07, 0.0, 0.35)
    wide[[1, 2]] = wide[[2, 1]]
    nbr = _tube_patch(0.18, 1.25, 0.05, 0.40)
    s = make_sheet(np.vstack([wide, nbr]),
                   [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6]])
    assert np.degrees(ix.curvature_span(s, ix.ring_matrix(s)).max()) > 30.0
    r = s.find_intersections()
    assert 0 not in set(int(x) for x in r.folded)


def test_small_faces_are_unaffected_by_the_curvature_guard():
    """A normal crypt-sized cell spans ~14 degrees and is judged normally."""
    ok = _tube_patch(0.0, 0.24, 0.0, 0.6)
    s = make_sheet(ok, [[0, 1, 2, 3]])
    assert np.degrees(ix.curvature_span(s, ix.ring_matrix(s)).max()) < 30.0
    assert len(s.find_intersections().folded) == 0

    bow = make_sheet(ok[[0, 1, 3, 2]], [[0, 1, 2, 3]])
    assert len(bow.find_intersections().folded) == 1


# ---------------------------------------------------------------------------
# Signed area screen
# ---------------------------------------------------------------------------
def test_signed_area_collapses_while_the_area_the_energy_sees_does_not():
    s = make_sheet(hexagon(), [[0, 1, 2, 3, 4, 5]])
    cache = ix.ring_matrix(s)
    signed, fan = ix.signed_area_defect(s, cache)
    assert np.allclose(abs(signed[0]), fan[0], rtol=1e-9)

    bad = make_sheet([[0, 0], [1, 0], [0, 1], [1, 1]], [[0, 1, 2, 3]])
    cache = ix.ring_matrix(bad)
    signed, fan = ix.signed_area_defect(bad, cache)
    assert abs(signed[0]) < 1e-12, "the lobes cancel exactly"
    assert fan[0] > 0.4, "but face area, being a sum of magnitudes, does not"


# ---------------------------------------------------------------------------
# check_intersections event
# ---------------------------------------------------------------------------
def test_behaviour_reappends_itself_and_reports():
    s = make_sheet([[0, 0], [1, 0], [0, 1], [1, 1]], [[0, 1, 2, 3]])
    manager = EventManager("face")
    manager.append(check_intersections)
    manager.update()
    manager.execute(s)

    report = s.settings["intersections"]
    assert len(report.folded) == 1
    assert any(f[0].__name__ == "check_intersections" for f in manager.next), \
        "must re-append itself so it runs every step"


def test_behaviour_history_and_raise():
    s = make_sheet([[0, 0], [1, 0], [0, 1], [1, 1]], [[0, 1, 2, 3]])
    history = []
    manager = EventManager("face")
    manager.append(check_intersections, history=history)
    manager.update()
    manager.execute(s)
    assert len(history) == 1 and len(history[0][1].folded) == 1

    manager = EventManager("face")
    manager.append(check_intersections, raise_on_detect=True)
    manager.update()
    with pytest.raises(IntersectionError):
        manager.execute(s)


def test_behaviour_is_silent_on_a_clean_sheet():
    s = make_sheet(hexagon(), [[0, 1, 2, 3, 4, 5]])
    manager = EventManager("face")
    manager.append(check_intersections, raise_on_detect=True)
    manager.update()
    manager.execute(s)                       # must not raise
    assert s.settings["intersections"].clean


def test_cache_is_rebuilt_when_topology_changes():
    """The ring matrix is reused between topology changes -- that is what makes a
    per-step call cheap -- but it must not go stale."""
    s = make_sheet(hexagon(), [[0, 1, 2, 3, 4, 5]])
    first = s.find_intersections()
    assert first.cache.n_faces == 1

    verts = np.vstack([hexagon(), hexagon(3.0, 0.0)])
    s2 = make_sheet(verts, [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]])
    second = s2.find_intersections(cache=first.cache)
    assert second.cache.n_faces == 2, "stale cache must be rebuilt"


def test_every_throttles_the_check():
    """`every` re-appends without running the detection, so a run can afford it."""
    s = make_sheet([[0, 0], [1, 0], [0, 1], [1, 1]], [[0, 1, 2, 3]])
    manager = EventManager("face")
    manager.append(check_intersections, every=5)
    for _ in range(4):
        manager.update()
        manager.execute(s)
    assert "intersections" in s.settings, "must run on the first step"
    ran_first = s.settings["intersections"]

    del s.settings["intersections"]
    manager.update()
    manager.execute(s)                      # step 4 -> skipped
    assert "intersections" not in s.settings
    assert ran_first is not None
    assert any(f[0].__name__ == "check_intersections" for f in manager.next)


def test_trigger_folded_still_finds_the_invaded_neighbour():
    """The cheap chained path must still name the victim when a face has folded."""
    a = [[-2, -0.15], [2, -0.15], [2, 0.15], [-2, 0.15]]
    s = make_sheet(a, [[0, 1, 3, 2]])       # swapped -> bow-tie
    assert len(s.find_intersections(trigger="folded").folded) == 1
