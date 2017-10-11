from ..utils import to_nd
import warnings


def model_factory(effectors):
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

        __doc__ = """Dynamical model with the following interactions:
{} from the effectors: {}.""".format(labels, effectors)

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
            if not hasattr(eptm, 'ucoords'):
                warnings.warn('setting ucoords in grad computation,'
                              'please fix your specs')
                eptm.ucoords = ['u' + c for c in eptm.coords]
                for uc in eptm.ucoords:
                    eptm.edge_df[uc] = 0.0

            eptm.edge_df[eptm.ucoords] = (
                eptm.edge_df[eptm.dcoords]
                / to_nd(eptm.edge_df['length'], eptm.dim))

            eptm.edge_df['is_active'] = (
                eptm.upcast_srce(eptm.vert_df['is_active'])
                * eptm.upcast_srce(eptm.face_df['is_alive']))

            grads = [f.gradient(eptm) for f in effectors]
            if components:
                return grads
            srce_grads = (g[0] for g in grads)
            trgt_grads = (g[1] for g in grads if g[1] is not None)
            grad_i = (eptm.sum_srce(sum(srce_grads))
                      + eptm.sum_trgt(sum(trgt_grads))
                      * to_nd(eptm.vert_df.is_active, eptm.dim))

            return grad_i / norm_factor

    return NewModel
