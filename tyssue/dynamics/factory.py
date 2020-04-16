import warnings
import numpy as np
from copy import deepcopy

from .effectors import dimensionalize as dimensionalize
from .effectors import normalize as normalize

from ..utils import to_nd


def model_factory(effectors, ref_effector=None):
    """Produces a Model class with the provided effectors.

    Parameters
    ----------
    effectors : list of :class:`.effectors.AbstractEffectors` classes.
    ref_effector : optional, default None
        if passed, will be used for normalization,
        by default, the last effector in the list is used

    Returns
    -------
    NewModel : a Model derived class with compute_enregy and compute_gradient
      methods

    """
    if ref_effector is None:
        ref_effector = effectors[-1]

    class NewModel:

        labels = []
        specs = {
            "cell": {},
            "face": {},
            "edge": {},
            "vert": {},
            "settings": {"nrj_norm_factor": 1.0},
        }

        _effectors = effectors

        for f in effectors:
            labels.append(f.label)
            try:
                for k in specs:
                    specs[k].update(f.specs.get(k, {}))
            except ValueError:
                warnings.warn(
                    """
Since 0.7, you need to provide a default value for each of the
specs parameters, e.g.
    specs = {
        "face": {
            "perimeter": 1.0,
            "perimeter_elasticity": 0.1,
            "prefered_perimeter": 3.81,
        }
    }

Setting all default values to 1.0 for now
"""
                )
                for k in specs:
                    specs[k].update({key: 1.0 for key in f.specs.get(k, {})})

        @staticmethod
        def dimensionalize(nondim_specs):
            dim_specs = deepcopy(nondim_specs)
            for effector in effectors:
                if effector == ref_effector:
                    continue
                dimensionalize(nondim_specs, dim_specs, effector, ref_effector)

            ref_nrj = ref_effector.get_nrj_norm(dim_specs)
            dim_specs["settings"]["nrj_norm_factor"] = ref_nrj
            return dim_specs

        @classmethod
        def dimentionalize(cls, nondim_specs):
            warnings.warn(
                """This badly worded method is deprecated,
 use dimensionalize instead"""
            )
            return cls.dimensionalize(nondim_specs)

        @staticmethod
        def normalize(dim_specs):
            nondim_specs = deepcopy(dim_specs)
            for effector in effectors:
                normalize(dim_specs, nondim_specs, effector, ref_effector)

        @staticmethod
        def compute_energy(eptm, full_output=False):
            energies = [f.energy(eptm) for f in effectors]
            norm_factor = eptm.specs["settings"].get("nrj_norm_factor", 1)
            if full_output:
                return [E / norm_factor for E in energies]

            return sum(E.sum() for E in energies) / norm_factor

        @staticmethod
        def compute_gradient(eptm, components=False):
            norm_factor = eptm.specs["settings"].get("nrj_norm_factor", 1)
            if not eptm.ucoords[0] in eptm.edge_df.columns:
                warnings.warn(
                    "setting ucoords in grad computation," "please fix your specs"
                )
                for uc in eptm.ucoords:
                    eptm.edge_df[uc] = 0.0

            eptm.edge_df[eptm.ucoords] = eptm.edge_df[eptm.dcoords] / to_nd(
                eptm.edge_df["length"], eptm.dim
            )

            grads = [f.gradient(eptm) for f in effectors]
            if components:
                return grads

            grad_s, grad_t, grad_v = None, None, None

            srce_grads = [g[0] for g in grads if g[0].shape[0] == eptm.Ne]
            if srce_grads:
                grad_s = eptm.sum_srce(sum(srce_grads))
            trgt_grads = [
                g[1] for g in grads if (g[1] is not None) and (g[1].shape[0] == eptm.Ne)
            ]
            if trgt_grads:
                grad_t = eptm.sum_trgt(sum(trgt_grads))
            vert_grads = [g[0] for g in grads if g[0].shape[0] == eptm.Nv]
            if vert_grads:
                grad_v = sum(vert_grads)

            grad_i = sum([g for g in (grad_s, grad_t, grad_v) if g is not None])

            return grad_i / norm_factor

    return NewModel
