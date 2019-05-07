import logging
import numpy as np
import pandas as pd
import warnings
from functools import wraps

from .intersection import self_intersections
from ..core.sheet import Sheet, get_outer_sheet

log = logging.getLogger(__name__)


def auto_collisions(fun):
    """Decorator to solve collisions detections after the
    execution of the decorated function.

    It is assumed that the two first arguments of the decorated
    function are a :class:`Sheet` object and a geometry class

    Note
    ----
    The function is re-executed with the updated geometry
    """

    @wraps(fun)
    def with_collision_correction(*args, **kwargs):
        log.debug("checking for collisions")
        eptm, geom = args[:2]
        eptm.position_buffer = eptm.vert_df[eptm.coords].copy()
        res = fun(*args, **kwargs)
        if isinstance(eptm, Sheet):
            change = solve_sheet_collisions(eptm, eptm.position_buffer)
        else:
            change = solve_bulk_collisions(eptm, eptm.position_buffer)
        if change:
            log.info("collision avoided")
        geom.update_all(eptm)
        return res

    return with_collision_correction


def solve_bulk_collisions(eptm, position_buffer):
    """Corrects the auto-collisions for the outer surface(s) of a 3D epithelium.

    Parameters
    ----------
    eptm : a :class:`Epithelium` object
    position_buffer : np.array of shape (eptm.Nv, eptm.dim):
        positions of the vertices prior to the collisions

    Returns
    -------
    changed : bool
        `True` if the positions of some vertices were changed

    """

    sub_sheet = get_outer_sheet(eptm)
    pos_idx = sub_sheet.vert_df.index.copy()
    sub_sheet.reset_index()
    sub_buffer = pd.DataFrame(
        position_buffer.loc[pos_idx].values,
        index=sub_sheet.vert_df.index,
        columns=sub_sheet.coords,
    )
    changed = solve_sheet_collisions(sub_sheet, sub_buffer)
    if changed:
        eptm.vert_df.loc[pos_idx, eptm.coords] = sub_sheet.vert_df[eptm.coords].values
    return changed


def solve_sheet_collisions(sheet, position_buffer):
    """Corrects the auto-collisions for the outer surface(s) of a 2.5D sheet.

    Parameters
    ----------
    sheet : a :class:`Sheet` object
    position_buffer : np.array of shape (sheet.Nv, sheet.dim):
        positions of the vertices prior to the collisions

    Returns
    -------
    changed : bool
        `True` if the positions of some vertices were changed

    """

    intersecting_edges = self_intersections(sheet)
    if intersecting_edges.shape[0]:
        log.info("%d intersections were detected", intersecting_edges.shape[0])
        shyness = sheet.settings.get("shyness", 1e-10)
        boxes = CollidingBoxes(sheet, position_buffer, intersecting_edges)
        changed = boxes.solve_collisions(shyness)
        return changed
    return False


class CollidingBoxes:
    """Utility class to manage collisions
    """

    def __init__(self, sheet, position_buffer, intersecting_edges):
        """Creates a CollidingBoxes instance

        Parameters
        ----------

        sheet : a :clas:`Sheet` instance
        position_buffer : np.array of shape (sheet.Nv, sheet.dim):
            positions of the vertices prior to the collisions
        intersecting_edges : np.ndarray
            pairs of indices of the intersecting edges

        """
        self.sheet = sheet
        self.edge_pairs = intersecting_edges
        self.face_pairs = self._get_intersecting_faces()
        self.edge_buffer = sheet.upcast_srce(position_buffer).copy()
        self.edge_buffer.columns = ["sx", "sy", "sz"]
        self.plane_not_found = False

    def _get_intersecting_faces(self):
        """Returns unique pairs of intersecting faces

        """
        _face_pairs = self.sheet.edge_df.loc[
            self.edge_pairs.flatten(), "face"
        ].values.reshape((-1, 2))
        unique_pairs = set(map(frozenset, _face_pairs))

        return np.array([[*pair] for pair in unique_pairs if len(pair) == 2])

    def get_limits(self, shyness=1e-10):
        """ Iterator over the position boundaries avoiding the
        collisions.

        Parameters
        ----------
        shyness : float
          the extra distance between two colliding vertices,
          on each side of the collision plane.

        Yields
        ------
        lower, upper : two `pd.Series`
           those Series are indexed by the vertices of the colliding
           faces giving the lower and upper bounds for the vertices
        """
        for face_pair in self.face_pairs:
            yield self._collision_plane(face_pair, shyness)

    def solve_collisions(self, shyness=1e-10):
        """ Solves the collisions by finding the collision plane.

        Modifies the sheet vertex positions inplace such that they
        rest at a distance ``shyness`` apart on each side of the collision plane.

        Parameters
        ----------
        shyness : float, default 1e-10
          the extra distance between two colliding vertices,
          on each side of the collision plane.

        Based on Liu, J.-D., Ko, M.-T., & Chang, R.-C. (1998),
        *A simple self-collision avoidance for cloth animation*.
        Computers & Graphics, 22(1), 117–128.
        `DOI <https://doi.org/doi:10.1016/s0097-8493(97)00087-3>`_

        """

        colliding_verts = self.sheet.edge_df[
            self.sheet.edge_df["face"].isin(self.face_pairs.ravel())
        ]["srce"].unique()
        upper_bounds = pd.DataFrame(
            index=pd.Index(colliding_verts, name="vert"), columns=self.sheet.coords
        )
        upper_bounds[:] = np.inf
        lower_bounds = pd.DataFrame(
            index=pd.Index(colliding_verts, name="vert"), columns=self.sheet.coords
        )
        lower_bounds[:] = -np.inf
        plane_found = False
        for lower, upper in self.get_limits(shyness):
            if lower is None:
                continue
            plane_found = True
            sub_lower = lower_bounds.loc[lower.index, lower.columns]
            lower_bounds.loc[lower.index, lower.columns] = np.maximum(sub_lower, lower)

            sub_upper = upper_bounds.loc[upper.index, upper.columns]
            upper_bounds.loc[upper.index, upper.columns] = np.minimum(sub_upper, upper)

        if upper_bounds.shape[0] == 0:
            plane_found = False

        if not plane_found:
            return False

        upper_bounds.index.name = "vert"
        upper_bounds = (
            upper_bounds[np.isfinite(upper_bounds.values.astype(float))]
            .groupby("vert")
            .apply(min)
        )
        lower_bounds.index.name = "vert"
        lower_bounds = (
            lower_bounds[np.isfinite(lower_bounds.values.astype(float))]
            .groupby("vert")
            .apply(max)
        )

        correction_upper = np.minimum(
            self.sheet.vert_df.loc[upper_bounds.index, self.sheet.coords],
            upper_bounds.values,
        )
        correction_lower = np.maximum(
            self.sheet.vert_df.loc[lower_bounds.index, self.sheet.coords],
            lower_bounds.values,
        )
        corrections = pd.concat((correction_lower, correction_upper), axis=0)
        self.sheet.vert_df.loc[
            corrections.index.values, self.sheet.coords
        ] = corrections
        return True

    def _collision_plane(self, face_pair, shyness):

        f0, f1 = face_pair

        fe0c = self.sheet.edge_df[self.sheet.edge_df["face"] == f0].copy()
        fe1c = self.sheet.edge_df[self.sheet.edge_df["face"] == f1].copy()
        fe0p = self.edge_buffer[self.sheet.edge_df["face"] == f0].copy()
        fe1p = self.edge_buffer[self.sheet.edge_df["face"] == f1].copy()

        bb0c = _face_bbox(fe0c)
        bb1c = _face_bbox(fe1c)
        bb0p = _face_bbox(fe0p)
        bb1p = _face_bbox(fe1p)

        dr0 = bb0c - bb0p
        dr1 = bb1c - bb1p
        sign_change_l1h0 = np.sign((bb1c.l - bb0c.h) * (bb1p.l - bb0p.h)) < 0
        sign_change_l0h1 = np.sign((bb0c.l - bb1c.h) * (bb0p.l - bb1p.h)) < 0

        # face 0 is to the left of face 1 on the collision axis
        if any(sign_change_l1h0):
            lower_bound = pd.DataFrame(index=fe1c.srce)
            upper_bound = pd.DataFrame(index=fe0c.srce)

            coll_ax = bb0c[sign_change_l1h0].index
            plane_coords = ((bb0p.h * dr1.l - bb1p.l * dr0.h) / (dr1.l - dr0.h)).loc[
                coll_ax
            ]
        # face 0 is to the right of face 1 on the collision axis
        elif any(sign_change_l0h1):
            lower_bound = pd.DataFrame(index=fe0c.srce)
            upper_bound = pd.DataFrame(index=fe1c.srce)
            coll_ax = bb0c[sign_change_l0h1].index
            plane_coords = ((bb1p.h * dr0.l - bb0p.l * dr1.h) / (dr0.l - dr1.h)).loc[
                coll_ax
            ]
        else:
            log.info("""Plane Not Found""")
            self.plane_not_found = True
            lower_bound = pd.DataFrame(
                index=set(fe0c.srce).union(fe1c.srce), columns=list("xyz")
            )
            upper_bound = pd.DataFrame(
                index=set(fe0c.srce).union(fe1c.srce), columns=list("xyz")
            )
            for c in list("xyz"):
                b0 = bb0c.loc[c]
                b1 = bb1c.loc[c]
                left, right = (fe0c, fe1c) if (b0.mean() < b1.mean()) else (fe1c, fe0c)

                lim = (left[f"s{c}"].max() + right[f"s{c}"].min()) / 2
                upper_bound.loc[right.srce, c] = right[f"s{c}"].max()
                upper_bound.loc[left.srce, c] = lim - shyness / 2

                lower_bound.loc[left.srce, c] = left[f"s{c}"].min()
                lower_bound.loc[right.srce, c] = lim + shyness / 2

            return lower_bound, upper_bound

        for c in coll_ax:
            lower_bound[c] = plane_coords.loc[c] + shyness / 2
            upper_bound[c] = plane_coords.loc[c] - shyness / 2

        return lower_bound, upper_bound


def _face_bbox(face_edges):

    points = face_edges[["sx", "sy", "sz"]].values
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    return pd.DataFrame(
        [lower, upper], index=list("lh"), columns=list("xyz"), dtype=np.float
    ).T


def revert_positions(sheet, position_buffer, intersecting_edges):

    unique_edges = np.unique(intersecting_edges)
    unique_verts = np.unique(sheet.edge_df.loc[unique_edges, ["srce", "trgt"]])
    sheet.vert_df.loc[unique_verts, sheet.coords] = position_buffer.loc[unique_verts]
