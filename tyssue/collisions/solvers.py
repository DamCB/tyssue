import logging
from functools import wraps

import numpy as np
import pandas as pd
import matplotlib.path as mplPath

from ..core.objects import _ordered_edges
from ..core.sheet import Sheet, get_outer_sheet
from ..core.monolayer import Monolayer
from .intersection import self_intersections

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
        solve_self_intersect_face(eptm)
        geom.update_all(eptm)
        if isinstance(eptm, Sheet):
            change = solve_sheet_collisions(eptm, eptm.position_buffer)
        else:
            change = solve_bulk_collisions(eptm, eptm.position_buffer)
        if change:
            log.info("collision avoided")
        geom.update_all(eptm)
        return res

    return with_collision_correction


def solve_self_intersect_face(eptm):
    face_self_intersect = eptm.edge_df.groupby("face").apply(_do_face_self_intersect)

    for f in face_self_intersect[face_self_intersect].index:
        sorted_edge = np.array(_ordered_edges(
            eptm.edge_df[eptm.edge_df["face"] == f][["srce", "trgt", "face"]])).flatten()[3::4]
        angle_list = np.arctan2(
            eptm.edge_df.loc[sorted_edge]["sy"].to_numpy() - eptm.edge_df.loc[sorted_edge]["fy"].to_numpy(),
            eptm.edge_df.loc[sorted_edge]["sx"].to_numpy() - eptm.edge_df.loc[sorted_edge][
                "fx"].to_numpy())

        angle_e = pd.DataFrame(angle_list, index=sorted_edge, columns=['angle'])

        if np.argmin(angle_e["angle"]) != 0:
            pos_s = np.argmin(angle_e["angle"])
            angle_e = pd.concat([angle_e.iloc[pos_s:], angle_e.iloc[:pos_s]])
            # Fix if the swap is between the 2 lowest angle value
            if ((angle_e.iloc[-1]["angle"] > angle_e.iloc[0]["angle"]) and (
                    angle_e.iloc[-1]["angle"] < angle_e.iloc[-2]["angle"])):
                pos_s = angle_e.shape[0] - 1
                angle_e = pd.concat([angle_e.iloc[pos_s:], angle_e.iloc[:pos_s]])

        angle_e = pd.concat([angle_e, angle_e.iloc[[0]]])
        angle_e.iloc[-1]["angle"] += 2 * np.pi

        pos_s = np.where(angle_e.diff()["angle"] < 0)[0][0]
        v1 = eptm.edge_df.loc[angle_e.index[pos_s]]["srce"]
        v2 = eptm.edge_df.loc[angle_e.index[pos_s - 1]]["srce"]

        v1_x, v1_y = eptm.vert_df.loc[v1][['x', 'y']]
        v2_x, v2_y = eptm.vert_df.loc[v2][['x', 'y']]
        eptm.vert_df.loc[v1, ['x', 'y']] = v2_x, v2_y
        eptm.vert_df.loc[v2, ['x', 'y']] = v1_x, v1_y


def _check_convexity(polygon):
    res = 0
    for i in range(polygon.shape[0] - 2):
        p = polygon[i]
        tmp = polygon[i + 1]
        v_x = polygon[i + 1][0] - polygon[i][0]
        v_y = polygon[i + 1][1] - polygon[i][1]
        u = polygon[i + 2]

        if i == 0:  # in first loop direction is unknown, so save it in res
            res = u[0] * v_y - u[1] * v_x + v_x * p[1] - v_y * p[0];
        else:
            newres = u[0] * v_y - u[1] * v_x + v_x * p[1] - v_y * p[0];
            if ((newres > 0 and res < 0) or (newres < 0 and res > 0)):
                return False

    return True


def _do_face_self_intersect(edge):
    sorted_edge = np.array(_ordered_edges(
        edge[['srce', 'trgt', 'face']])).flatten()[3::4]
    angle_list = np.arctan2(
        edge.loc[sorted_edge]['sy'].to_numpy() - edge.loc[sorted_edge]['fy'].to_numpy(),
        edge.loc[sorted_edge]['sx'].to_numpy() - edge.loc[sorted_edge][
            'fx'].to_numpy())

    angle_e = pd.DataFrame(angle_list, index=sorted_edge, columns=['angle'])

    if np.argmin(angle_e['angle']) != 0:
        pos_s = np.argmin(angle_e['angle'])
        angle_e = pd.concat([angle_e.iloc[pos_s:], angle_e.iloc[:pos_s]])

    angle_e = pd.concat([angle_e, angle_e.iloc[[0]]])
    angle_e.iloc[-1]['angle'] += 2 * np.pi

    if (not pd.Series(angle_e['angle']).is_monotonic_increasing) and (
            _check_convexity(edge.loc[sorted_edge][['sx', 'sy']].to_numpy())):
        return True
    return False


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
        if len(sheet.coords) == 2:
            boxes = CollidingBoxes2D(sheet, position_buffer, intersecting_edges)
        elif len(sheet.coords) == 3:
            boxes = CollidingBoxes3D(sheet, position_buffer, intersecting_edges)
        changed = boxes.solve_collisions(shyness)
        return changed
    return False


class CollidingBoxes:
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
        self.edge_buffer.columns = ["s" + p for p in self.edge_buffer.columns]
        self.plane_not_found = False

    def solve_collisions(self, shyness=1e-10):
        return True

    def _get_intersecting_faces(self):
        """Returns unique pairs of intersecting faces"""
        _face_pairs = self.sheet.edge_df.loc[
            self.edge_pairs.flatten(), "face"
        ].values.reshape((-1, 2))
        unique_pairs = set(map(frozenset, _face_pairs))

        return np.array([[*pair] for pair in unique_pairs if len(pair) == 2])


class CollidingBoxes2D(CollidingBoxes):
    def __init__(self, sheet, position_buffer, intersecting_edges):
        CollidingBoxes.__init__(self, sheet, position_buffer, intersecting_edges)

    def _find_vert_inside(self, edge1, edge2):
        triangle1 = [self.sheet.edge_df.loc[edge1][['sx', 'sy']].to_numpy(),
                     self.sheet.edge_df.loc[edge1][['tx', 'ty']].to_numpy(),
                     self.sheet.edge_df.loc[edge1][['fx', 'fy']].to_numpy()]
        triangle2 = [self.sheet.edge_df.loc[edge2][['sx', 'sy']].to_numpy(),
                     self.sheet.edge_df.loc[edge2][['tx', 'ty']].to_numpy(),
                     self.sheet.edge_df.loc[edge2][['fx', 'fy']].to_numpy()]

        if _point_in_triangle(self.sheet.edge_df.loc[edge1][['sx', 'sy']].to_numpy(), triangle2):
            return self.sheet.edge_df.loc[edge1]['srce'], self.sheet.edge_df.loc[edge2]['face'], edge2
        if _point_in_triangle(self.sheet.edge_df.loc[edge1][['tx', 'ty']].to_numpy(), triangle2):
            return self.sheet.edge_df.loc[edge1]['trgt'], self.sheet.edge_df.loc[edge2]['face'], edge2
        if _point_in_triangle(self.sheet.edge_df.loc[edge2][['sx', 'sy']].to_numpy(), triangle1):
            return self.sheet.edge_df.loc[edge2]['srce'], self.sheet.edge_df.loc[edge1]['face'], edge1
        if _point_in_triangle(self.sheet.edge_df.loc[edge2][['tx', 'ty']].to_numpy(), triangle1):
            return self.sheet.edge_df.loc[edge2]['trgt'], self.sheet.edge_df.loc[edge1]['face'], edge1

        # search inside full face
        if _point_in_polygon(self.sheet, self.sheet.edge_df.loc[edge1][['sx', 'sy']].to_numpy(),
                             self.sheet.edge_df.loc[edge2]['face']):
            return self.sheet.edge_df.loc[edge1]['srce'], self.sheet.edge_df.loc[edge2]['face'], edge2
        if _point_in_polygon(self.sheet, self.sheet.edge_df.loc[edge1][['tx', 'ty']].to_numpy(),
                             self.sheet.edge_df.loc[edge2]['face']):
            return self.sheet.edge_df.loc[edge1]['trgt'], self.sheet.edge_df.loc[edge2]['face'], edge2
        if _point_in_polygon(self.sheet, self.sheet.edge_df.loc[edge2][['sx', 'sy']].to_numpy(),
                             self.sheet.edge_df.loc[edge1]['face']):
            return self.sheet.edge_df.loc[edge2]['srce'], self.sheet.edge_df.loc[edge1]['face'], edge1
        if _point_in_polygon(self.sheet, self.sheet.edge_df.loc[edge2][['tx', 'ty']].to_numpy(),
                             self.sheet.edge_df.loc[edge1]['face']):
            return self.sheet.edge_df.loc[edge2]['trgt'], self.sheet.edge_df.loc[edge1]['face'], edge1

        return np.NaN, np.NaN, np.NaN

    def solve_collisions(self, shyness=1e-10):
        id_vert_change = []
        for e1, e2 in self.edge_pairs:
            # Dont fix if crossing occur between two neighboring cells or between "2 same" cell.
            if (self.sheet.edge_df.loc[e1]['face'] != self.sheet.edge_df.loc[e2]['face']) and (
                    self.sheet.edge_df.loc[e1]['face'] not in self.sheet.get_neighbors(
                    self.sheet.edge_df.loc[e2]['face'])):
                vertices = self.sheet.edge_df.loc[[e1, e2]][['srce', 'trgt']].to_numpy().flatten()
                if vertices.all() not in id_vert_change:
                    vert_inside, face, edge = self._find_vert_inside(e1, e2)
                    if not np.isnan(vert_inside):
                        if vert_inside not in id_vert_change:
                            new_pos = _line_intersection(
                                (self.sheet.vert_df.loc[vert_inside][['x', 'y']],
                                 self.sheet.face_df.loc[face][['x', 'y']]),
                                (
                                    self.sheet.edge_df.loc[edge][['sx', 'sy']],
                                    self.sheet.edge_df.loc[edge][['tx', 'ty']]))
                            if not np.isnan(new_pos[0]):
                                self.sheet.vert_df.loc[vert_inside, self.sheet.coords] = new_pos
                                id_vert_change.append(vert_inside)
        return True


class CollidingBoxes3D(CollidingBoxes):
    """Utility class to manage collisions"""

    def __init__(self, sheet, position_buffer, intersecting_edges):
        CollidingBoxes.__init__(self, sheet, position_buffer, intersecting_edges)

    def get_limits(self, shyness=1e-10):
        """Iterator over the position boundaries avoiding the
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
        """Solves the collisions by finding the collision plane.

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
        ## Need to be corrected
        # Move vertex too far away
        # Move vertex that shouldn't be moved
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
                index=set(fe0c.srce).union(fe1c.srce), columns=self.sheet.coords
            )
            upper_bound = pd.DataFrame(
                index=set(fe0c.srce).union(fe1c.srce), columns=self.sheet.coords
            )
            for c in self.sheet.coords:
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

    def _face_bbox(self, face_edges):
        """
        Get the minimal box that contain the face
        """
        if "sz" in face_edges.columns:
            points = face_edges[["sx", "sy", "sz"]].values
        else:
            points = face_edges[["sx", "sy"]].values
        lower = points.min(axis=0)
        upper = points.max(axis=0)
        return pd.DataFrame(
            [lower, upper], index=list("lh"), columns=self.coord[:len(lower)], dtype=float
        ).T


def _point_in_triangle(point, triangle):
    """Returns True if the point is inside the triangle
    and returns False if it falls outside.
    - The argument *point* is a tuple with two elements
    containing the X,Y coordinates respectively.
    - The argument *triangle* is a tuple with three elements each
    element consisting of a tuple of X,Y coordinates.

    It works like this:
    Walk clockwise or counterclockwise around the triangle
    and project the point onto the segment we are crossing
    by using the dot product.
    Finally, check that the vector created is on the same side
    for each of the triangle's segments.
    """
    # Unpack arguments
    x, y = point
    ax, ay = triangle[0]
    bx, by = triangle[1]
    cx, cy = triangle[2]
    # Segment A to B
    side_1 = (x - bx) * (ay - by) - (ax - bx) * (y - by)
    # Segment B to C
    side_2 = (x - cx) * (by - cy) - (bx - cx) * (y - cy)
    # Segment C to A
    side_3 = (x - ax) * (cy - ay) - (cx - ax) * (y - ay)
    # All the signs must be positive or all negative
    return (side_1 < 0.0) == (side_2 < 0.0) == (side_3 < 0.0)


def _point_in_polygon(sheet, point, face):
    edge_index = sheet.edge_df[sheet.edge_df['face'] == face].index
    ordered_vert_index = np.array(_ordered_edges(sheet.edge_df.loc[edge_index][['srce', 'trgt', 'face']])).flatten()[
                         0::4]
    poly_path = mplPath.Path(np.array(sheet.vert_df.loc[ordered_vert_index][['x', 'y']]))
    #     point = np.array(sheet.edge_df.loc[8][['tx', 'ty']])
    return poly_path.contains_point(point)


def _line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        #         raise Exception('lines do not intersect')
        return np.NaN, np.NaN

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def revert_positions(sheet, position_buffer, intersecting_edges):
    unique_edges = np.unique(intersecting_edges)
    unique_verts = np.unique(sheet.edge_df.loc[unique_edges, ["srce", "trgt"]])
    sheet.vert_df.loc[unique_verts, sheet.coords] = position_buffer.loc[unique_verts]
