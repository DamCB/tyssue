import warnings

from copy import deepcopy

from .effectors import dimensionalize as dimensionalize
from .effectors import normalize as normalize

from ..utils import to_nd


def model_factory(effectors, ref_effector):
    """Produces a Model class with the provided effectors.

    Parameters
    ----------
    effectors : list of :class:`.effectors.AbstractEffectors` classes.

    Returns
    -------
    NewModel : a Model derived class with compute_enregy and compute_gradient
      methods

    """

    class NewModel:

        labels = []
        specs = {'cell': set(),
                 'face': set(),
                 'edge': set(),
                 'vert': set()}
        for f in effectors:
            labels.append(f.label)
            specs[f.element] = specs[f.element].union(f.specs)

        __doc__ = """Dynamical model with the following effectors:\n"""
        __doc__ = __doc__+'\n'.join(labels)

        @staticmethod
        def dimensionalize(nondim_specs):
            dim_specs = deepcopy(nondim_specs)
            for effector in effectors:
                if effector == ref_effector:
                    continue
                dimensionalize(nondim_specs, dim_specs,
                               effector, ref_effector)

            ref_nrj = ref_effector.get_nrj_norm(dim_specs)
            dim_specs['settings']['nrj_norm_factor'] = ref_nrj
            return dim_specs

        @classmethod
        def dimentionalize(cls, nondim_specs):
            warnings.warn('''This badly worded method is deprecated,
 use dimensionalize instead''')
            return cls.dimensionalize(nondim_specs)

        @staticmethod
        def normalize(dim_specs):
            nondim_specs = deepcopy(dim_specs)
            for effector in effectors:
                normalize(dim_specs, nondim_specs,
                          effector, ref_effector)

        @staticmethod
        def compute_energy(eptm, full_output=False):
            energies = [f.energy(eptm) for f in effectors]
            norm_factor = eptm.specs['settings'].get('nrj_norm_factor', 1)
            if full_output:
                return [E / norm_factor for E in energies]

            return sum(E.sum() for E in energies) / norm_factor

        @staticmethod
        def compute_gradient(eptm, components=False):
            norm_factor = eptm.specs['settings'].get('nrj_norm_factor', 1)
            if not eptm.ucoords[0] in eptm.edge_df.columns:
                warnings.warn('setting ucoords in grad computation,'
                              'please fix your specs')
                for uc in eptm.ucoords:
                    eptm.edge_df[uc] = 0.0

            eptm.edge_df[eptm.ucoords] = (
                eptm.edge_df[eptm.dcoords]
                / to_nd(eptm.edge_df['length'], eptm.dim))

            eptm.edge_df['is_active'] = (
                eptm.upcast_srce(eptm.vert_df['is_active'])
                * eptm.upcast_face(eptm.face_df['is_alive']))

            grads = [f.gradient(eptm) for f in effectors]
            if components:
                return grads
            srce_grads = (g[0] for g in grads
                          if g[0].shape[0] == eptm.Ne)
            trgt_grads = (g[1] for g in grads
                          if (g[1] is not None)
                          and (g[1].shape[0] == eptm.Ne))
            vert_grads = (g[0] for g in grads
                          if g[0].shape == eptm.Nv)

            grad_i = (eptm.sum_srce(sum(srce_grads))
                      + eptm.sum_trgt(sum(trgt_grads))
                      + sum(vert_grads)) * to_nd(eptm.vert_df.is_active,
                                                 eptm.dim)

            return grad_i / norm_factor

    return NewModel
