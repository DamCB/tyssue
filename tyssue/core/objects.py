"""
Core definitions
"""
import logging
import warnings
from collections import deque
from copy import deepcopy
from itertools import product

import numpy as np
import pandas as pd

from ..geometry.planar_geometry import PlanarGeometry
from ..geometry.sheet_geometry import SheetGeometry
from ..utils import connectivity
from ..utils.utils import set_data_columns, spec_updater

log = logging.getLogger(name=__name__)


class Epithelium:
    """Base class defining a connective tissue in 2D or 3D."""

    def __init__(self, identifier, datasets, specs=None, coords=None, maxbackup=5):
        """Creates an epithelium

        Parameters
        ----------
        identifier : string
        datasets : dictionary of dataframes
            The keys correspond to the different geometrical elements
            constituting the epithelium:

            * `vert` contains a dataframe of vertices,
            * `edge` contains a dataframe of *oriented* half-edges between vertices,
            * `face` contains a dataframe of polygonal faces enclosed by half-edges,
            * `cell` contains a dataframe of polyhedral cells delimited by faces,

        specs : nested dictionnary of specifications
            The first key designs the element name: (`vert`, `edge`, `face`, `cell`),
            corresponding to the respective dataframes attribute in the dataset.
            The second level keys design column names of the above dataframes.
            For exemple:
            .. code::
                specs = {
                    "face": {
                        ## Face Geometry
                        "perimeter": 0.,
                        "area": 0.,
                        ## Coordinates
                        "x": 0.,
                        "y": 0.,
                        "z": 0.,
                        ## Topology
                        "num_sides": 6,
                        ## Masks
                        "is_alive": 1},
                    "vert": {
                        ## Coordinates
                        "x": 0.,
                        "y": 0.,
                        "z": 0.,
                        ## Masks
                        "is_active": 1},
                    "edge": {
                        ## Connections
                        "srce": 0,
                        "trgt": 1,
                        "face": 0,
                        "cell": 0,
                        ## Coordinates
                        "dx": 0.,
                        "dy": 0.,
                        "dz": 0.,
                        "length": 0.,
                        ## Normals
                        "nx": 0.,
                        "ny": 0.,
                        "nz": 1.}
                    "settings":
                        ## Custom values
                        "geometry": "flat"
                    }

        Note
        ----
        For efficiency reasons, we have to maintain a monotonous RangeIndex
        for each dataset. Thus, **the index of an element can change**,
        and should not be used as an identifier.

        """
        # backup container
        # TODO: pass the max backup number as a config argument
        self._backups = deque(maxlen=maxbackup)

        # each of those has a separate dataframe, as well as entries in
        # the specification files
        _frame_types = {"edge", "vert", "face", "cell"}
        self.identifier = identifier
        if not set(datasets).issubset(_frame_types):
            raise ValueError(
                "The `datasets` dictionnary should"
                " contain keys in {}".format(_frame_types)
            )
        self.datasets = datasets
        self.data_names = list(datasets.keys())
        self.element_names = ["srce", "trgt", "face", "cell"][: len(self.data_names)]
        # Infer specs from the first rows of the datasets
        if specs is None:
            specs = {elem: df.iloc[0].to_dict() for elem, df in datasets.items()}
        if "settings" not in specs:
            specs["settings"] = {}

        self.specs = specs
        if coords is None:
            coords = [c for c in "xyz" if c in datasets["vert"].columns]

        # Add upcast coordinates to specs
        self.specs["edge"].update(
            {
                e + c: 0.0
                for e, c in product("dustfc"[: len(self.data_names) + 2], coords)
            }
        )
        self.specs["face"].update({c: 0.0 for c in coords})
        self.update_specs(specs, reset=False)

        self.coords = coords
        # edge's dx, dy, dz
        self.dcoords = ["d" + c for c in self.coords]
        # edge's unit length vector
        self.ucoords = ["u" + c for c in self.coords]

        self.dim = len(self.coords)
        # edge's normals
        if self.dim == 3:
            self.ncoords = ["n" + c for c in self.coords]
            self.update_specs({"edge": {nc: 0.0 for nc in self.ncoords}}, reset=False)
        else:
            self.update_specs({"edge": {"nz": 0.0}}, reset=False)

        self._bad = None
        self.bbox = None
        if "is_active" in self.vert_df.columns:
            self.active_verts = self.vert_df[self.vert_df.is_active == 1].index
        else:
            self.active_verts = self.vert_df.index
        self.set_bbox()

        self.position_buffer = None
        self.topo_changed = False
        self.is_ordered = False

    @property
    def vert_df(self):
        """The face :class:`pd.DataFrame` containing vertex associated
        data e.g. the position of each vertex.
        """
        return self.datasets["vert"]

    @vert_df.setter
    def vert_df(self, value):
        self.datasets["vert"] = value

    @property
    def face_df(self):
        """The face :class:`pd.DataFrame` containing face associated
        data e.g. the position of their center or their area
        """
        return self.datasets["face"]

    @face_df.setter
    def face_df(self, value):
        self.datasets["face"] = value

    @property
    def edge_df(self):
        """The edge :class:`pd.DataFrame` containing edge associated
        data e.g. their length.This dataframe also contains the whole
        connexion of the epithelium through the `"srce", "trgt", "face", "cell"`
        indices. In 2D, a half-edge is associated with a single (face, srce, trgt)
        positively oriented triangle. In 3D, the (cell, face, srce, trgt)
        positively oriented terahedron is also unique.
        """
        return self.datasets["edge"]

    @edge_df.setter
    def edge_df(self, value):
        self.datasets["edge"] = value

    @property
    def cell_df(self):
        """The cell :class:`pd.DataFrame` containing cell associated
        data e.g. the position of their center or their volume
        """
        return self.datasets.get("cell", None)

    @cell_df.setter
    def cell_df(self, value):
        self.datasets["cell"] = value

    def copy(self, deep_copy=True):
        """
        Returns a copy of the epithelium

        Parameters
        ----------
        deep_copy : bool, default True
            if True, use a copy of the original object's datasets
            to create the new object. If False, datasets are not copied
        """
        if deep_copy:
            datasets = {element: df.copy() for element, df in self.datasets.items()}
            specs = deepcopy(self.specs)

        else:  # pragma: no cover
            log.info("New epithelium object from %s without deep copy", self.identifier)
            datasets = self.datasets
            specs = self.specs

        identifier = self.identifier + "_copy"

        new = type(self)(identifier, datasets, specs=specs, coords=self.coords)

        return new

    def backup(self):
        """Creates a copy of self and keeps a reference to it
        in the self._backups deque.

        """
        log.info("Backing up")
        self._backups.append(self.copy(deep_copy=True))

    def restore(self):
        """Resets the eptithelium data to its last backed up state

        A copy of the current state prior to restoring is kept in the
        `_bad` attribute for inspection.
        Calling this method multiple times (without calling backup) will
        go back in the epithelium backups.
        """

        log.info("Restoring")
        log.info(
            "a copy of the unrestored epithelium is stored in the `_bad` attribute"
        )
        bck = self._backups.pop()
        self._bad = self.copy(deep_copy=True)
        self.datasets = bck.datasets
        self.specs = bck.specs

    @property
    def settings(self):
        """Accesses the `specs['settings']` dictionnary."""
        return self.specs["settings"]

    def update_specs(self, new, reset=False):
        """Recursively updates the `self.specs` nested dictionnary,
        and set the new values to the corresponding columns
        in the datasets. If reset is `True`, existing values
        will be overwritten.
        """
        spec_updater(self.specs, new)
        set_data_columns(self.datasets, new, reset)

    def update_num_sides(self):
        """Updates the number of half-edges around the faces.
        The data is registered in the `"num_sides"` column of
        `self.face_df`.
        """
        self.face_df["num_sides"] = self.edge_df.face.value_counts()

    def update_num_faces(self):
        """Updates the number of faces around the cells.
        The data is registered in the `"num_faces"` column of
        `self.cell_df`.
        """
        self.cell_df["num_faces"] = self.edge_df.groupby("cell").apply(
            lambda df: df["face"].unique().size
        )
        self.cell_df["num_ridges"] = self.edge_df.cell.value_counts()

    def update_rank(self):
        st_connect = connectivity.srce_trgt_connectivity(self)
        self.vert_df["rank"] = ((st_connect + st_connect.T) > 0).sum(axis=0)

    def reset_topo(self):
        """Recomputes the number of sides for the faces and the
        number of faces for the cells.
        """
        log.debug("Resetting topology")
        self.update_num_sides()
        if "is_active" in self.vert_df.columns:
            self.active_verts = self.vert_df[self.vert_df.is_active == 1].index
        if "cell" in self.data_names:
            self.update_num_faces()

    @property
    def Nv(self):
        """The number of vertices in the epithelium."""
        return self.vert_df.shape[0]

    @property
    def Ne(self):
        """The number of edges in the epithelium."""
        return self.edge_df.shape[0]

    @property
    def Nf(self):
        """The number of faces in the epithelium."""
        return self.face_df.shape[0]

    @property
    def Nc(self):
        """The number of cells in the epithelium."""
        if "cell" not in self.data_names:
            return self.face_df.shape[0]
        return self.cell_df.shape[0]

    def _upcast(self, idx, df):

        # Assumes a flat index
        upcast = df.take(idx, axis=0)
        try:
            upcast.index = self.edge_df.index
        except AttributeError:
            if len(upcast.shape) == 1:
                upcast = pd.Series(upcast, index=self.edge_df.index)
            else:
                upcast = pd.DataFrame(upcast, index=self.edge_df.index)
        return upcast

    def upcast_cols(self, element, columns):
        """Syntactic sugar to upcast from the epithelium datasets.

        Parameters
        ----------
        element: {'srce'|'trgt'|'face'|'cell'}
           corresponding self.edge_df column over which to index
           if element is 'srce' or 'trgt', the upcast data will be
           taken form self.vert_df
        columns: index
           the column(s) to be taken from the input dataset.

        """
        if element in ["srce", "trgt"]:
            dataset = "vert"
        else:
            dataset = element
        return self._upcast(self.edge_df[element], self.datasets[dataset][columns])

    def upcast_srce(self, df):
        """Reindexes input data to self.edge_df.index
        by repeating the values for each source entry.

        Parameters
        ----------
        df : :class:`pd.DataFrame`, :class:`pd.Series` :class:`np.ndarray` or string
          The data to be upcasted. If array like, should have `self.Nv` elements.
          If a string is passed it should be a column of `self.vert_df`

        Returns
        -------
        upcast_df : :class:`pd.DataFrame` or :class:`pd.Series`
          The value repeated like the values of `self.edge_df["srce"]`
        """
        if isinstance(df, str):
            df = self.vert_df[df]
        return self._upcast(self.edge_df["srce"], df)

    def upcast_trgt(self, df):
        """Reindexes input data to self.edge_df.index
        by repeating the values for each target entry

        Parameters
        ----------
        df : :class:`pd.DataFrame`, :class:`pd.Series` :class:`np.ndarray` or string
          The data to be upcasted. If array like, should have `self.Nv` elements.
          If a string is passed it should be a column of `self.vert_df`

        Returns
        -------
        upcast_df : :class:`pd.DataFrame` or :class:`pd.Series`
          The value repeated like the values of `self.edge_df["trgt"]`
        """
        if isinstance(df, str):
            df = self.vert_df[df]
        return self._upcast(self.edge_df["trgt"], df)

    def upcast_face(self, df):
        """Reindexes input data to self.edge_df.index
        by repeating the values for each face entry

        Parameters
        ----------
        df : :class:`pd.DataFrame`, :class:`pd.Series` :class:`np.ndarray` or string
          The data to be upcasted. If array like, should have `self.Nf` elements.
          If a string is passed it should be a column of `self.face_df`

        Returns
        -------
        upcast_df : :class:`pd.DataFrame` or :class:`pd.Series`
          The value repeated like the values of `self.edge_df["face"]`

        """
        if isinstance(df, str):
            df = self.face_df[df]
        return self._upcast(self.edge_df["face"], df)

    def upcast_cell(self, df):
        """Reindexes input data to self.edge_df.index
        by repeating the values for each cell entry

        Parameters
        ----------
        df : :class:`pd.DataFrame`, :class:`pd.Series` :class:`np.ndarray` or string
          The data to be upcasted. If array like, should have `self.Nc` elements.
          If a string is passed it should be a column of `self.cell_df`

        Returns
        -------
        upcast_df : :class:`pd.DataFrame` or :class:`pd.Series`
          The value repeated like the values of `self.edge_df["cell"]`
        """
        if isinstance(df, str):
            df = self.cell_df[df]
        return self._upcast(self.edge_df["cell"], df)

    def _lvl_sum(self, df, lvl):
        df_ = df
        if isinstance(df, np.ndarray):
            df_ = pd.DataFrame(df, index=self.edge_df.index)
        elif isinstance(df, pd.Series):
            df_ = df.to_frame()
        elif lvl not in df.columns:
            df_ = df.copy()
        df_[lvl] = self.edge_df[lvl]
        return df_.groupby(lvl).sum()

    def sum_srce(self, df):
        """Sums the values of the edge-indexed dataframe `df` grouped by
        the values of `self.edge_df["srce"]`

        Returns
        -------
        summed : :class:`pd.DataFrame` the summed data, indexed by the source vertices.
        """
        return self._lvl_sum(df, "srce")

    def sum_trgt(self, df):
        """Sums the values of the edge-indexed dataframe `df` grouped by
        the values of `self.edge_df["trgt"]`

        Returns
        -------
        summed : :class:`pd.DataFrame` the summed data, indexed by the source vertices.
        """
        return self._lvl_sum(df, "trgt")

    def sum_face(self, df):
        """Sums the values of the edge-indexed dataframe `df` grouped by
        the values of `self.edge_df["face"]`

        Returns
        -------
        summed : :class:`pd.DataFrame` the summed data, indexed by the source vertices.
        """
        return self._lvl_sum(df, "face")

    def sum_cell(self, df):
        """Sums the values of the edge-indexed dataframe `df` grouped by
        the values of `self.edge_df["cell"]`

        Returns
        -------
        summed : :class:`pd.DataFrame` the summed data, indexed by the source vertices.
        """
        return self._lvl_sum(df, "cell")

    def get_orbits(self, center, periph):
        """Returns a dataframe with a `(center, edge)` MultiIndex with `periph`
        elements.

        Parameters
        ----------
        center : str,
            the name of the center element for example 'face', 'srce'
        periph : str,
            the name of the periphery elements, for example 'trgt', 'cell'

        Example
        -------
        >>> cell_verts = sheet.get_orbits('face', 'srce')
        >>> cell_verts.loc[45]
        edge
        218    75
        219    78
        220    76
        221    81
        222    90
        223    87
        Name: srce, dtype: int64

        """
        orbits = self.edge_df.groupby(center).apply(lambda df: df[periph])
        return orbits

    def idx_lookup(self, elem_id, element):
        """returns the current index of the element with the `"id"` column equal to `elem_id`

        Parameters
        ----------
        elem_id : int
          id of the element to retrieve
        element : {"vert"|"edge"|"face"|"cell"}
          the corresponding dataset.
        """
        df = self.datasets[element]["id"]
        idx = df[df == elem_id].index
        if len(idx):
            return idx[0]
        else:
            return None

    def get_neighbors(self, elem_id, elem="cell"):
        """Returns the indexes of the adjacent elements (cells or faces) of
        the element of index `elem_id`.

        Parameters
        ----------
        elem_id : int
            the index of the central element (a face or a cell)
        element : {'cell' | 'face'}, default 'cell'

        Returns
        -------
        neghbors : set
            the cells (or faces) sharing an edge with the central cell (face)
        """

        topo = self.edge_df[["srce", "trgt", elem]]
        edges = self.edge_df[self.edge_df[elem] == elem_id][["srce", "trgt"]]

        neighbors = set(
            topo[
                (topo["srce"].isin(edges["srce"]) & topo["trgt"].isin(edges["trgt"]))
                | (topo["srce"].isin(edges["trgt"]) & topo["trgt"].isin(edges["srce"]))
            ][elem]
        )
        return neighbors - {elem_id}

    def get_neighborhood(self, elem_id, order, elem="cell"):
        """Returns `elem_id` neighborhood up to a degree of `order`

        For example, if `order` is 2, it wil return the adjacent cells (or faces)
        and theses cells neighbors.

        Returns
        -------
        neighbors : pd.DataFrame with two colums, the index
            of the neighboring cell (face), and it's neighboring order

        """

        neighbors = pd.DataFrame.from_dict({elem: [elem_id], "order": [0]})

        for k in range(order + 1):
            for neigh in neighbors[neighbors["order"] == k - 1][elem]:
                new_neighs = self.get_neighbors(neigh, elem)
                new_neighs = set(new_neighs).difference(neighbors[elem])
                orders = np.ones(len(new_neighs), dtype=int) * (k)
                new_neighs = pd.DataFrame.from_dict(
                    {elem: list(new_neighs), "order": orders}, dtype=int
                )
                neighbors = pd.concat([neighbors, new_neighs])
        return neighbors.reset_index(drop=True).loc[1:]

    def face_polygons(self, coords=None):
        """Returns a pd.Series of arrays with the coordinates the face polygons

        Each element of the Series is a (num_sides, num_dims) array of points
        ordered counterclockwise.

        Note
        ----
        Vertices are assumed to be ordered in a face. If you are not
        sure it is the case, you can run `sheet.reset_index(order=True)` before calling
        this function.
        """
        if not self.is_ordered:
            raise ValueError(
                "The vertices are assumed to be correctly ordered around the cell"
            )
        if coords is None:
            coords = self.coords

        scoords = ["s" + c for c in coords]
        if not set(scoords).issubset(self.edge_df.columns):
            for c in coords:
                self.edge_df["s" + c] = self.upcast_srce(self.vert_df[c])

        polys = self.edge_df.groupby("face").apply(lambda df: df[scoords].to_numpy())
        return polys

    def validate(self):
        """returns True if the mesh is validated

        e.g. has only closed polygons and polyhedra
        """
        return np.alltrue(self.get_valid())

    def get_valid(self):
        """Set the 'is_valid' column to true if the faces are all closed polygons,
        and the cells closed polyhedra.
        """
        is_valid_face = self.edge_df.groupby("face").apply(_test_valid)
        is_valid = self.upcast_face(is_valid_face)
        if "cell" in self.data_names:
            is_valid_cell = self.edge_df.groupby("cell").apply(_is_closed_cell)
            is_valid = np.logical_and(is_valid, self.upcast_cell(is_valid_cell))
        self.edge_df["is_valid"] = is_valid
        return is_valid

    def get_invalid(self):
        """Returns a mask over self.edge_df for invalid faces."""
        is_valid = self.get_valid()
        return ~is_valid

    def sanitize(self, trim_borders=False, order_edges=False):
        """Removes invalid faces and associated vertices

        If trim_borders is True (defaults to False), there will be a single
        border edge per border face.
        """
        invalid_edges = self.get_invalid()
        if not any(invalid_edges) and trim_borders:
            from ..topology.base_topology import merge_border_edges

            merge_border_edges(self)
            self.topo_changed = False
            return

        self.remove(invalid_edges, trim_borders, order_edges)
        self.topo_changed = False

    def remove(self, edge_out, trim_borders=False, order_edges=False):
        """Remove the edges indexed by `edge_out` associated with all
        the cells and faces containing those edges

        If trim_borders is True (defaults to False), there will be a single
        border edge per border face.
        """
        top_level = self.element_names[-1]
        log.info("Removing cells at the %s level", top_level)
        fto_rm = self.edge_df.loc[edge_out, top_level].unique()
        if not fto_rm.shape[0]:
            log.info("Nothing to remove")
            return
        if fto_rm.shape[0] == self.datasets[top_level].shape[0]:
            raise ValueError("sanitize would delete the whole epithlium")

        fto_rm.sort()
        log.info("%d %s level elements will be removed", len(fto_rm), top_level)

        edge_df_ = (
            self.edge_df.set_index(top_level, append=True).swaplevel(0, 1).sort_index()
        )
        to_rm = np.concatenate([edge_df_.loc[c].index.values for c in fto_rm])
        to_rm.sort()
        self.edge_df = self.edge_df.drop(to_rm)

        remaining_verts = np.unique(self.edge_df[["srce", "trgt"]])
        self.vert_df = self.vert_df.loc[remaining_verts]
        if top_level == "face":
            self.face_df = self.face_df.drop(fto_rm)
        elif top_level == "cell":
            remaining_faces = np.unique(self.edge_df["face"])
            self.face_df = self.face_df.loc[remaining_faces]
            self.cell_df = self.cell_df.drop(fto_rm)
        self.reset_index()
        self.reset_topo()
        if trim_borders:
            from ..topology.base_topology import merge_border_edges

            merge_border_edges(self)
        if order_edges:
            self.reset_index(order=True)

    def cut_out(self, bbox, coords=None):
        """Returns the index of edges with at least one vertex outside of the bounding box

        Parameters
        ----------
        bbox : sequence of shape (dim, 2)
             the bounding box as (min, max) pairs for each coordinates.
        coords : list of str of len dim, default None
             the coords corresponding to the bbox.
        """
        if coords is None:
            coords = self.coords
        outs = pd.DataFrame(index=self.edge_df.index, columns=coords)
        for c, bounds in zip(coords, bbox):
            out_vert_ = (self.vert_df[c] < bounds[0]) | (self.vert_df[c] > bounds[1])
            outs[c] = self.upcast_srce(out_vert_) | self.upcast_trgt(out_vert_)

        edge_out = outs.sum(axis=1).astype(bool)
        return self.edge_df[edge_out].index

    def set_bbox(self, margin=0.0):
        """Sets the attribute `bbox` with pairs of values bellow
        and above the min and max of the vert coords, with a margin.
        """
        self.bbox = np.array(
            [
                [self.vert_df[c].min() - margin, self.vert_df[c].max() + margin]
                for c in self.coords
            ]
        )

    def reset_index(self, order=False):
        """Resets the datasets  to have continuous indices

        If order is True (the default), sorts the edges
        such that for each face, vertices are ordered clockwize
        """
        log.debug("reseting index for %s", self.identifier)
        self.topo_changed = True
        # remove disconnected vertices and faces
        self.vert_df = self.vert_df.reindex(
            set(self.edge_df.srce).union(self.edge_df.trgt)
        )
        self.face_df = self.face_df.reindex(set(self.edge_df.face))

        new_vidx = pd.Series(np.arange(self.vert_df.shape[0]), index=self.vert_df.index)

        self.edge_df["srce"] = new_vidx.reindex(self.edge_df["srce"]).values.astype(int)
        self.edge_df["trgt"] = new_vidx.reindex(self.edge_df["trgt"]).values.astype(int)

        new_fidx = pd.Series(np.arange(self.face_df.shape[0]), index=self.face_df.index)

        self.edge_df["face"] = new_fidx.loc[self.edge_df["face"]].values.astype(int)

        self.vert_df.reset_index(drop=True, inplace=True)
        self.vert_df.index.name = "vert"

        self.face_df.reset_index(drop=True, inplace=True)
        self.face_df.index.name = "face"

        if "cell" in self.data_names:
            self.cell_df = self.cell_df.reindex(set(self.edge_df["cell"]))
            new_cidx = pd.Series(
                np.arange(self.cell_df.shape[0]), index=self.cell_df.index
            )
            self.edge_df["cell"] = new_cidx.loc[self.edge_df["cell"]].values
            self.cell_df.reset_index(drop=True, inplace=True)
            self.cell_df.index.name = "cell"

        if order:
            if self.dim == 2:
                phis = PlanarGeometry.get_phis(self)
            else:
                if "rx" not in self.edge_df:
                    SheetGeometry.update_dcoords(self)
                    SheetGeometry.update_centroid(self)
                phis = SheetGeometry.get_phis(self)
            self.edge_df["phi"] = phis
            self.edge_df.sort_values(["face", "phi"], inplace=True)
            self.is_ordered = True
        else:
            self.is_ordered = False

        self.edge_df.reset_index(drop=True, inplace=True)
        self.edge_df.index.name = "edge"

    def triangular_mesh(self, coords=None, return_mask=False):
        """
        Return a triangulation of an epithelial sheet (2D in a 3D space),
        with added edges between face barycenters and junction vertices.

        Parameters
        ----------
        coords : list of str:
          pair of coordinates corresponding to column names
          for self.face_df and self.vert_df

        return_mask : bool, optional, default True
          if True, returns `face_mask`

        Returns
        -------
        vertices : (self.Nf+self.Nv, 3) ndarray
           all the vertices' coordinates
        triangles : (self.Ne, 3) ndarray of ints
           triple of the vertices' indexes forming
           the triangular elements. For each junction edge, this is simply
           the index (srce, trgt, face). This is correctly oriented.
        face_mask: (self.Nf + self.Nv,) mask with 1 iff the vertex corresponds
           to a face center
        """
        if coords is None:
            coords = self.coords

        vertices = np.concatenate((self.face_df[coords], self.vert_df[coords]), axis=0)

        # edge indices as (Nf + Nv) * 3 array
        triangles = self.edge_df[["srce", "trgt", "face"]].values
        # The src, trgt, face triangle is correctly oriented
        # both vert_idx cols are shifted by Nf
        triangles[:, :2] += self.Nf

        # Ensure returned arrays are C-contiguous
        if not vertices.data.c_contiguous:
            vertices = np.array(vertices, order="C")

        if not triangles.data.c_contiguous:
            triangles = np.array(triangles, order="C")

        if not return_mask:
            return vertices, triangles

        face_mask = np.arange(self.Nf + self.Nv) < self.Nf
        return vertices, triangles, face_mask

    def vertex_mesh(self, coords, vertex_normals=True):
        """Returns the vertex coordinates and a list of vertex indices
        for each face of the tissue.
        If `vertex_normals` is True, also returns the normals of each vertex
        (set as the average of the vertex' edges), suitable for .OBJ export

        Note
        ----
        Vertices are assumed to be ordered in a face. If you are not
        sure it is the case, you can run `sheet.reset_index()` before calling
        this function.

        """
        vertices = self.vert_df[coords]
        faces = self.edge_df.groupby("face").apply(lambda df: list(df["srce"]))
        faces = faces.dropna()

        if vertex_normals:
            normals = (
                self.edge_df.groupby("srce")[self.ncoords].mean()
                + self.edge_df.groupby("trgt")[self.ncoords].mean()
            ) / 2.0
            return vertices.to_numpy(), faces.to_numpy(), normals.to_numpy()
        return vertices.to_numpy(), faces.to_numpy()

    def validate_closed_cells(self):
        """Returns True if all cells of the epithelium are closed."""
        euler_chars = self.edge_df.groupby("cell").apply(euler_characteristic)
        return np.array_equal(np.unique(euler_chars), 2)

    def get_opposite_faces(self):
        """Populates the 'opposite' column of self.face_df with the index of
        the opposite face or -1 if the face has no opposite.

        """
        face_v = self.edge_df.groupby("face").apply(lambda df: frozenset(df["srce"]))
        face_v2 = pd.Series(data=face_v.index, index=face_v.values)
        grouped = face_v2.groupby(level=0)
        cardinal = grouped.apply(len)
        if cardinal.max() > 2:
            raise ValueError(
                "Invalid topology, the following faces have more than one neighbor: "
                f"{face_v2[cardinal > 2].to_list()}"
            )
        self.face_df["opposite"] = -1

        face_pairs = (
            face_v2[cardinal == 2]
            .groupby(level=0)
            .apply(lambda df: dict(enumerate(df)))
            .values.reshape((-1, 2))
        )
        if not face_pairs.shape[0]:
            return
        self.face_df.loc[face_pairs[:, 0], "opposite"] = face_pairs[:, 1]
        self.face_df.loc[face_pairs[:, 1], "opposite"] = face_pairs[:, 0]


def get_opposite_faces(eptm):
    warnings.warn("Deprecated, use `eptm.get_opposite_faces()` instead")
    eptm.get_opposite_faces()


def _ordered_edges(face_edges):
    """Returns "srce", "trgt" and "face" indices
    organized clockwise for each face.

    Parameters
    ----------
    face_edges: `pd.DataFrame`
        exerpt of an edge_df for a single face

    Returns
    -------
    edges: list of 3 ints
        srce, trgt, face indices, ordered
    """
    srces, trgts, faces = face_edges[["srce", "trgt", "face"]].values.T
    srce, trgt, face_ = srces[0], trgts[0], faces[0]
    edges = [[srce, trgt, face_]]
    for face_ in faces[1:]:
        srce, trgt = trgt, trgts[srces == trgt][0]
        edges.append([srce, trgt, face_])
    return edges


def _ordered_vert_idxs(face):
    try:
        return [idxs[0] for idxs in _ordered_edges(face)]
    except IndexError:
        return np.nan


def get_next_edges(sheet):
    """
    returns a pd.Series with the index of the next
    edge for each edge
    """
    next_e = sheet.edge_df.groupby("face").apply(_next_edge)
    next_e.index = next_e.index.droplevel("face")
    return next_e.sort_index()


def get_prev_edges(sheet):
    """
    returns a pd.Series with the index of the next
    edge for each edge
    """
    prev_e = sheet.edge_df.groupby("face").apply(_prev_edge)
    prev_e.index = prev_e.index.droplevel("face")
    return prev_e.sort_index()


def get_simple_index(edge_df):
    """
    returns a subset of the edge_df index corresponding
    to the non oriented edges (aka full edges).

    This is faster than `get_extra_indices` and works also in 3D

    """
    srted = np.sort(edge_df[["srce", "trgt"]].to_numpy(), axis=1)
    shift = np.ceil(np.log10(edge_df.srce.max()))
    multi = int(10 ** (shift))
    st_hash = srted[:, 0] * multi + srted[:, 1]
    st_hash = pd.Series(st_hash, index=edge_df.index)
    return st_hash.drop_duplicates().index.values


def euler_characteristic(edge_df):
    """Returns the Euler characteristic of the (non oriented) mesh represented by edge_df.

    The Euler characteristic is
    the number of vertices minus the number of edges plus the number of faces

    It is equal to 2 for a closed-on-itself mesh (topologicaly eq. to a sphere),
    1 to a mesh with a border. It is not unique for monoloyers or bulk epithelia
    but provides a way to check wether a cell is closed.

    """
    V = edge_df["srce"].unique().shape[0]
    F = edge_df["face"].unique().shape[0]
    E = get_simple_index(edge_df).shape[0]
    return V - E + F


def _next_edge(edf):

    edf["edge"] = edf.index
    next_edge = edf.set_index("srce", append=False).loc[edf["trgt"], "edge"].values
    return pd.Series(index=edf.index, data=next_edge)


def _prev_edge(edf):

    edf["edge"] = edf.index
    next_edge = edf.set_index("trgt", append=False).loc[edf["srce"], "edge"].values
    return pd.Series(index=edf.index, data=next_edge)


def _is_closed_cell(e_df):
    return euler_characteristic(e_df) == 2


def _test_invalid(face):
    """Returns True if the source and target sets of the faces polygon
    are different or if the face polygon is not closed
    """

    s1 = set(face["srce"])
    s2 = set(face["trgt"])
    if s1 != s2:
        return True
    ordered = np.array(_ordered_edges(face))
    if not np.all(ordered[:, 0] == np.roll(ordered[:, 1], 1)):
        return True
    return False


def _test_valid(face):
    """Returns true iff all sources are also targets for the faces polygon."""
    return np.logical_not(_test_invalid(face))
