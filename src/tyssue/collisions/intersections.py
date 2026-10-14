"""Detect faces of an epithelium that fold or overlap.

  Test A  self-intersection  -- a face's own boundary crosses itself (a "bow-tie")
  Test B  containment        -- a vertex not on face F's ring lies inside F
  Test C  overlap            -- two faces share interior points
  plus    open rings         -- a face whose half-edges do not close into a cycle

Each face is tested in the best-fit plane of its own ring, so curved and
periodic sheets need no special handling. Faces too curved for a plane to
represent are reported as *undecidable*. Candidate pairs are culled with each
face's bounding ball, so the tests are cheap enough to run every step.

See Also
--------
:func:`tyssue.behaviors.sheet.basic_events.check_intersections`
    run the tests during a simulation
:meth:`tyssue.core.objects.Epithelium.find_intersections`
"""

from __future__ import annotations

from dataclasses import dataclass, field

import logging

import numpy as np
from scipy.spatial import cKDTree

log = logging.getLogger(__name__)

__all__ = [
    "curvature_span",
    "RingCache",
    "ring_matrix",
    "face_frames",
    "self_intersecting",
    "contained_vertices",
    "overlapping_faces",
    "find_intersections",
    "Result",
    "signed_area_defect",
]


# ---------------------------------------------------------------------------
# Topology: the padded ring matrix
# ---------------------------------------------------------------------------
@dataclass
class RingCache:
    """Face rings as dense padded arrays, reusable until the topology changes.

    RING  (Nf, nmax) int   positional row index into the vertex coordinate array
    LEN   (Nf,)      int   sides per face
    VALID (Nf, nmax) bool  RING slot is a real vertex, not padding
    NXT   (Nf, nmax) int   slot of the next ring vertex, wrapping at LEN
    """

    RING: np.ndarray
    LEN: np.ndarray
    VALID: np.ndarray
    NXT: np.ndarray
    face_ids: np.ndarray
    vert_ids: np.ndarray
    signature: tuple = field(default=(), repr=False)
    open_rings: np.ndarray = field(default=None, repr=False)

    @property
    def n_faces(self) -> int:
        return len(self.LEN)


def _topology_signature(eptm) -> tuple:
    return (int(eptm.Nf), int(eptm.Ne), int(eptm.Nv))


def ring_matrix(eptm, cache: RingCache | None = None) -> RingCache:
    """Chain each face's half-edges ``srce -> trgt`` into a closed ring.

    Faces whose half-edges do not close into a single cycle are flagged in
    ``open_rings`` and excluded from the tests. `cache` is returned unchanged
    if the topology has not changed.
    """
    sig = _topology_signature(eptm)
    if cache is not None and cache.signature == sig:
        return cache

    edge = eptm.edge_df
    face_col = edge["face"].to_numpy()
    srce = edge["srce"].to_numpy()
    trgt = edge["trgt"].to_numpy()

    order = np.argsort(face_col, kind="stable")
    face_sorted = face_col[order]
    face_ids, starts = np.unique(face_sorted, return_index=True)
    bounds = np.append(starts, len(face_sorted))

    rings, open_flags = [], []
    for k in range(len(face_ids)):
        sl = order[bounds[k]:bounds[k + 1]]
        s, t = srce[sl], trgt[sl]
        nxt = dict(zip(s.tolist(), t.tolist()))
        ok = len(nxt) == len(s)
        ring = []
        if ok:
            start = int(s[0])
            ring = [start]
            cur = nxt[start]
            while cur != start:
                ring.append(int(cur))
                cur = nxt.get(cur)
                if cur is None or len(ring) > len(s):
                    ok = False
                    break
            ok = ok and len(ring) == len(s)
        if not ok:                      # open ring: placeholder, flagged
            ring = [int(s[0])] * max(len(s), 3)
        rings.append(ring)
        open_flags.append(not ok)

    nmax = max(len(r) for r in rings)
    nf = len(rings)
    RING = np.zeros((nf, nmax), dtype=np.int64)
    LEN = np.array([len(r) for r in rings], dtype=np.int64)
    for i, r in enumerate(rings):
        RING[i, :len(r)] = r
    k = np.arange(nmax)[None, :]
    VALID = k < LEN[:, None]
    NXT = np.where(VALID, (k + 1) % np.maximum(LEN[:, None], 1), 0)

    vert_ids = eptm.vert_df.index.to_numpy()
    lookup = np.full(int(vert_ids.max()) + 1, -1, dtype=np.int64)
    lookup[vert_ids] = np.arange(len(vert_ids))
    RING = np.where(VALID, lookup[np.clip(RING, 0, None)], 0)

    return RingCache(RING, LEN, VALID, NXT, face_ids, vert_ids, sig,
                     np.array(open_flags, dtype=bool))


# ---------------------------------------------------------------------------
# Geometry: per-face tangent frame
# ---------------------------------------------------------------------------
def _positions(eptm) -> np.ndarray:
    """(Nv, 3) vertex coordinates; 2D epithelia are lifted to z = 0."""
    coords = list(getattr(eptm, "coords", None) or
                  [c for c in ("x", "y", "z") if c in eptm.vert_df.columns])
    P = eptm.vert_df[coords].to_numpy(dtype=float)
    if P.shape[1] == 2:
        P = np.column_stack([P, np.zeros(len(P))])
    return P


def face_frames(eptm, cache: RingCache, pos: np.ndarray | None = None):
    """Per-face centroid, Newell area vector and an orthonormal tangent frame.

    Returns (c, N, u, w, P2, D, fan_area): P2 is the ring projected into (u, w),
    D holds the local differences p_i - c_F, and fan_area is tyssue's own face area
    (the sum of |n_i|/2 over half-edges).
    """
    P = _positions(eptm) if pos is None else pos
    ring = P[cache.RING]                                   # (Nf, nmax, 3)
    ring = np.where(cache.VALID[..., None], ring, 0.0)
    c = ring.sum(axis=1) / cache.LEN[:, None]

    D = np.where(cache.VALID[..., None], ring - c[:, None, :], 0.0)
    rows = np.arange(cache.n_faces)[:, None]
    Dn = D[rows, cache.NXT]
    tri = np.cross(D, Dn)                                  # per-fan-triangle normals
    N = tri.sum(axis=1)                                    # Newell area vector

    # The frame comes from the best-fit plane, not the Newell vector, which
    # vanishes on a symmetric bow-tie. Newell only sets the orientation.
    cov = np.einsum("fki,fkj->fij", D, D)
    _, vecs = np.linalg.eigh(cov)
    nh = vecs[:, :, 0]                                     # least-variance direction
    flip = np.einsum("fi,fi->f", nh, N) < 0
    nh = np.where(flip[:, None], -nh, nh)

    ref = np.tile(np.array([1.0, 0.0, 0.0]), (len(nh), 1))
    ref[np.abs(nh[:, 0]) > 0.9] = np.array([0.0, 1.0, 0.0])
    u = np.cross(nh, ref)
    un = np.linalg.norm(u, axis=1, keepdims=True)
    u = np.divide(u, un, out=np.zeros_like(u), where=un > 0)
    w = np.cross(nh, u)

    P2 = np.stack([np.einsum("fkd,fd->fk", D, u),
                   np.einsum("fkd,fd->fk", D, w)], axis=-1)
    # same as the face area of SheetGeometry: sum of |n_i| / 2
    fan_area = 0.5 * np.where(cache.VALID, np.linalg.norm(tri, axis=2), 0.0).sum(axis=1)
    return c, N, u, w, P2, D, fan_area


def signed_area_defect(eptm, cache: RingCache, pos=None):
    """Signed area of each face in its own plane, and its unsigned area.

    A folded face has a signed area smaller than its unsigned area. This is
    only a quick screen: a bow-tie with unequal lobes is missed, use
    :func:`self_intersecting` to detect folds.
    """
    _, N, _, _, P2, _, fan_area = face_frames(eptm, cache, pos)
    a, b = P2[..., 0], P2[..., 1]
    rows = np.arange(cache.n_faces)[:, None]
    an, bn = a[rows, cache.NXT], b[rows, cache.NXT]
    cross = np.where(cache.VALID, a * bn - an * b, 0.0)
    signed = 0.5 * cross.sum(axis=1)
    return signed, fan_area



def curvature_span(eptm, cache: RingCache, pos=None):
    """Angle (radians) the surface normal turns across each face.

    Each vertex gets the mean normal of its faces; a face's span is the largest
    angle between the normals of its vertices. A planar projection of the face
    is only meaningful when this is small.
    """
    P = _positions(eptm) if pos is None else pos
    _, _, u, w, _, _, _ = face_frames(eptm, cache, P)
    n = np.cross(u, w)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)

    acc = np.zeros((len(cache.vert_ids), 3))
    flat = cache.RING[cache.VALID]
    rep = np.repeat(np.arange(cache.n_faces), cache.LEN)
    np.add.at(acc, flat, n[rep])
    acc /= np.maximum(np.linalg.norm(acc, axis=1, keepdims=True), 1e-30)

    vn = acc[cache.RING]                                  # (Nf, nmax, 3)
    vn = np.where(cache.VALID[..., None], vn, np.nan)
    dots = np.einsum("fid,fjd->fij", np.nan_to_num(vn), np.nan_to_num(vn))
    ok = cache.VALID[:, :, None] & cache.VALID[:, None, :]
    dots = np.where(ok, np.clip(dots, -1.0, 1.0), 1.0)
    return np.arccos(dots.min(axis=(1, 2)))


# ---------------------------------------------------------------------------
# Test A -- self-intersection
# ---------------------------------------------------------------------------
def _proper_cross(p1, p2, p3, p4):
    """Segments (p1,p2) and (p3,p4) cross at an interior point of both.

    Touching or collinear segments, such as adjacent ring edges, do not count.
    """
    d1 = (p4[..., 0] - p3[..., 0]) * (p1[..., 1] - p3[..., 1]) - \
         (p4[..., 1] - p3[..., 1]) * (p1[..., 0] - p3[..., 0])
    d2 = (p4[..., 0] - p3[..., 0]) * (p2[..., 1] - p3[..., 1]) - \
         (p4[..., 1] - p3[..., 1]) * (p2[..., 0] - p3[..., 0])
    d3 = (p2[..., 0] - p1[..., 0]) * (p3[..., 1] - p1[..., 1]) - \
         (p2[..., 1] - p1[..., 1]) * (p3[..., 0] - p1[..., 0])
    d4 = (p2[..., 0] - p1[..., 0]) * (p4[..., 1] - p1[..., 1]) - \
         (p2[..., 1] - p1[..., 1]) * (p4[..., 0] - p1[..., 0])
    return (d1 * d2 < 0) & (d3 * d4 < 0)


def self_intersecting(eptm, cache: RingCache, pos=None, P2=None,
                      max_span_rad: float = np.deg2rad(30.0), return_span=False):
    """Test A. Boolean over faces: does this face's boundary cross itself?

    Faces spanning more than `max_span_rad` of curvature are excluded and, with
    `return_span`, returned as a separate `undecidable` mask with the spans.
    """
    if P2 is None:
        _, _, _, _, P2, _, _ = face_frames(eptm, cache, pos)
    nf, nmax, _ = P2.shape
    rows = np.arange(nf)
    LEN, NXT = cache.LEN, cache.NXT
    hit = np.zeros(nf, dtype=bool)

    for i in range(nmax):
        for j in range(i + 2, nmax):
            live = (i < LEN) & (j < LEN)
            if not live.any():
                continue
            i2 = NXT[rows, i]
            j2 = NXT[rows, j]
            # non-adjacent: neither edge may end where the other begins
            live &= (j2 != i) & (i2 != j)
            if not live.any():
                continue
            hit |= live & _proper_cross(P2[rows, i], P2[rows, i2],
                                        P2[rows, j], P2[rows, j2])
    hit &= ~cache.open_rings

    # a crossing found in a plane that cannot represent the face is not reliable
    span = curvature_span(eptm, cache, pos)
    undecidable = hit & (span > max_span_rad)
    hit &= ~undecidable
    if return_span:
        return hit, undecidable, span
    return hit


# ---------------------------------------------------------------------------
# Test B -- a foreign vertex inside a face
# ---------------------------------------------------------------------------
def _winding_inside(poly: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Nonzero-winding point-in-polygon for one polygon and many query points.

    Unlike the even-odd rule, a doubly-wound lobe of a folded face counts as
    inside.
    """
    n = len(poly)
    x, y = poly[:, 0], poly[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    qx = q[:, 0][:, None]
    qy = q[:, 1][:, None]

    up = (y[None, :] <= qy) & (yn[None, :] > qy)
    dn = (y[None, :] > qy) & (yn[None, :] <= qy)
    # side of the directed edge the query point falls on
    side = (xn - x)[None, :] * (qy - y[None, :]) - (yn - y)[None, :] * (qx - x[None, :])
    wind = np.where(up & (side > 0), 1, 0) + np.where(dn & (side < 0), -1, 0)
    return wind.sum(axis=1) != 0


def contained_vertices(eptm, cache: RingCache, pos=None, faces=None,
                       slab: float = 0.25):
    """Test B. Vertices that are not on face F's ring but lie inside F.

    Returns a list of (face_id, vert_id, owner_face_ids).

    Only vertices within `slab * sqrt(area)` of F's plane are tested, which
    allows for the curvature of the sheet.
    """
    P = _positions(eptm) if pos is None else pos
    c, N, u, w, P2, _, fan_area = face_frames(eptm, cache, P)

    radius = np.linalg.norm(P2, axis=2)
    radius = np.where(cache.VALID, radius, 0.0).max(axis=1)
    eps = slab * np.sqrt(np.maximum(fan_area, 1e-30))

    tree = cKDTree(P)
    todo = np.arange(cache.n_faces) if faces is None else np.asarray(faces)
    todo = todo[~cache.open_rings[todo]]

    out = []
    vert2face = None
    for f in todo:
        cand = tree.query_ball_point(c[f], radius[f])
        if not cand:
            continue
        own = set(cache.RING[f, :cache.LEN[f]].tolist())
        cand = np.array([j for j in cand if j not in own], dtype=np.int64)
        if not len(cand):
            continue
        D = P[cand] - c[f]
        h = D @ np.cross(u[f], w[f])
        near = np.abs(h) <= eps[f]
        if not near.any():
            continue
        cand = cand[near]
        D = D[near]
        q2 = np.column_stack([D @ u[f], D @ w[f]])
        poly = P2[f, :cache.LEN[f]]
        inside = _winding_inside(poly, q2)
        if not inside.any():
            continue
        if vert2face is None:
            vert2face = _vertex_to_faces(cache)
        for j in cand[inside]:
            vid = int(cache.vert_ids[j])
            owners = sorted(int(cache.face_ids[k]) for k in vert2face.get(j, ()))
            out.append((int(cache.face_ids[f]), vid, owners))
    return out


def _vertex_to_faces(cache: RingCache) -> dict:
    m = {}
    for f in range(cache.n_faces):
        for k in cache.RING[f, :cache.LEN[f]]:
            m.setdefault(int(k), set()).add(f)
    return m


# ---------------------------------------------------------------------------
# Test C -- complete pairwise overlap
# ---------------------------------------------------------------------------
def overlapping_faces(eptm, cache: RingCache, pos=None):
    """Test C. Face pairs whose boundaries properly cross.

    Catches "lens" overlaps where no vertex of either face lies inside the
    other, which Test B misses. Slower than A and B; run by
    ``find_intersections(strict=True)``.
    """
    P = _positions(eptm) if pos is None else pos
    c, _, u, w, P2, _, _ = face_frames(eptm, cache, P)
    radius = np.where(cache.VALID, np.linalg.norm(P2, axis=2), 0.0).max(axis=1)

    tree = cKDTree(c)
    pairs = tree.query_ball_tree(tree, r=2 * radius.max())
    out, indeterminate = [], []
    for f, others in enumerate(pairs):
        if cache.open_rings[f]:
            continue
        ringf = cache.RING[f, :cache.LEN[f]]
        polyf = P2[f, :cache.LEN[f]]
        for g in others:
            if g <= f or cache.open_rings[g]:
                continue
            if np.linalg.norm(c[f] - c[g]) > radius[f] + radius[g]:
                continue
            ringg = cache.RING[g, :cache.LEN[g]]
            if set(ringf.tolist()) & set(ringg.tolist()):
                continue                      # neighbours share an edge legitimately

            # project both rings onto the best-fit plane of the pair
            pts = np.vstack([P[ringf], P[ringg]])
            cc = pts.mean(axis=0)
            dd = pts - cc
            _, vecs = np.linalg.eigh(dd.T @ dd)
            nh = vecs[:, 0]
            ax = np.array([1.0, 0.0, 0.0]) if abs(nh[0]) <= 0.9 else np.array([0.0, 1.0, 0.0])
            uu = np.cross(nh, ax)
            uu /= np.linalg.norm(uu)
            ww = np.cross(nh, uu)

            # too curved for any plane to represent the pair
            resid = np.abs(dd @ nh).max()
            scale = max(radius[f], radius[g])
            if resid > 0.35 * scale:
                indeterminate.append((int(cache.face_ids[f]), int(cache.face_ids[g])))
                continue

            nf_, ng_ = len(ringf), len(ringg)
            proj = np.column_stack([dd @ uu, dd @ ww])
            if _pair_crosses(proj[:nf_], proj[nf_:]):
                out.append((int(cache.face_ids[f]), int(cache.face_ids[g])))
    if indeterminate:
        log.debug("%d face pairs too non-planar to decide", len(indeterminate))
    return out


def _pair_crosses(A: np.ndarray, B: np.ndarray) -> bool:
    An, Bn = np.roll(A, -1, axis=0), np.roll(B, -1, axis=0)
    for i in range(len(A)):
        if _proper_cross(A[i][None, :], An[i][None, :], B, Bn).any():
            return True
    return False


# ---------------------------------------------------------------------------
# Entry point, also available as Epithelium.find_intersections
# ---------------------------------------------------------------------------
@dataclass
class Result:
    folded: np.ndarray                 # face ids whose own boundary self-crosses
    contained: list                    # (face_id, vert_id, owner_face_ids)
    overlaps: list                     # (face_id, face_id), strict mode only
    open_rings: np.ndarray             # face ids whose half-edges do not close
    undecidable: np.ndarray = None     # crossings in a plane too curved to trust
    cache: RingCache = field(repr=False, default=None)

    @property
    def clean(self) -> bool:
        return not (len(self.folded) or len(self.contained)
                    or len(self.overlaps) or len(self.open_rings)
                    or (self.undecidable is not None and len(self.undecidable)))

    def __len__(self):
        return len(self.folded) + len(self.contained) + len(self.overlaps)


def find_intersections(eptm, cache: RingCache | None = None, *, strict: bool = False,
                       trigger: str | None = None, slab: float = 0.25) -> Result:
    """Run the intersection tests on an epithelium.

    strict   also run Test C (pairwise overlap), the slowest test.
    trigger  "folded" runs Test B only on faces adjacent to a self-crossing
             one. Faster, but misses convex cells sliding through each other.
    slab     see :func:`contained_vertices`
    """
    cache = ring_matrix(eptm, cache)
    P = _positions(eptm)
    _, _, _, _, P2, _, _ = face_frames(eptm, cache, P)

    folded, undecidable, _span = self_intersecting(eptm, cache, P, P2,
                                                   return_span=True)

    if trigger == "folded":
        if folded.any():
            v2f = _vertex_to_faces(cache)
            near = set()
            for f in np.flatnonzero(folded):
                for k in cache.RING[f, :cache.LEN[f]]:
                    near |= v2f.get(int(k), set())
            subset = np.array(sorted(near), dtype=np.int64)
        else:
            subset = np.array([], dtype=np.int64)
    else:
        subset = None

    contained = ([] if (trigger == "folded" and subset is not None and not len(subset))
                 else contained_vertices(eptm, cache, P, subset, slab))
    overlaps = overlapping_faces(eptm, cache, P) if strict else []

    return Result(
        undecidable=cache.face_ids[undecidable],
        folded=cache.face_ids[folded],
        contained=contained,
        overlaps=overlaps,
        open_rings=cache.face_ids[cache.open_rings],
        cache=cache,
    )
