import numpy as np
import pandas as pd
from scipy.interpolate import Rbf

from .sheet import Sheet


class MultiSheet:
    def __init__(self, name, layer_datasets, specs):

        self.coords = ["x", "y", "z"]
        self.layers = [
            Sheet("{}_{}".format(name, i), dsets, specs, coords=self.coords)
            for i, dsets in enumerate(layer_datasets)
        ]
        for i, layer in enumerate(self):
            for dset in layer.datasets.values():
                dset["layer"] = i

    def __iter__(self):
        for layer in self.layers:
            yield layer

    def __getitem__(self, n):
        return self.layers[n]

    def __len__(self):
        return len(self.layers)

    @property
    def Nes(self):
        return [layer.Ne for i, layer in self]

    @property
    def Nvs(self):
        return [layer.Nv for i, layer in self]

    @property
    def Nfs(self):
        return [layer.Nf for i, layer in self]

    @property
    def v_idxs(self):
        return np.array([sheet.Nv for sheet in self]).cumsum()

    @property
    def f_idxs(self):
        return np.array([sheet.Nf for sheet in self]).cumsum()

    @property
    def e_idxs(self):
        return np.array([sheet.Ne for sheet in self]).cumsum()

    def concat_datasets(self):
        datasets = {}

        v_dfs = [self[0].vert_df]
        e_dfs = [self[0].edge_df]
        f_dfs = [self[0].face_df]

        v_shift = 0
        f_shift = 0
        e_shift = 0
        for lower, upper in zip(self[:-1], self[1:]):
            v_shift += lower.Nv
            v_dfs.append(upper.vert_df.set_index(upper.vert_df.index + v_shift))
            f_shift += lower.Nf
            f_dfs.append(upper.face_df.set_index(upper.face_df.index + f_shift))
            e_shift += lower.Ne
            shifted_edge_df = upper.edge_df.set_index(upper.edge_df.index + e_shift)
            shifted_edge_df[["srce", "trgt"]] += v_shift
            shifted_edge_df["face"] += f_shift
            e_dfs.append(shifted_edge_df)

        for key, dfs in zip(["edge", "face", "vert"], [e_dfs, f_dfs, v_dfs]):
            datasets[key] = pd.concat(dfs)
        return datasets

    def update_interpolants(self):

        self.interpolants = [
            Rbf(
                sheet.vert_df["x"],
                sheet.vert_df["y"],
                sheet.vert_df["z"],
                **sheet.specs["settings"]["interpolate"]
            )
            for sheet in self
        ]
        # for interp in self.interpolants:
        #     interp.nodes = interp.nodes.clip(-1e2, 1e2)
