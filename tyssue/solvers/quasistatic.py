"""Quasistatic solver for vertex models

"""
import numpy as np

from scipy import optimize
from .. import config
from ..collisions import solve_collisions
from ..topology.sheet_topology import auto_t1, auto_t3


class QSSolver:
    """Quasistatic solver performing a gradient descent on a :class:`tyssue.Epithelium`
    object.

    Methods
    -------
    find_energy_min : energy minimization calling `scipy.optimize.minimize`
    approx_grad : uses `optimize.approx_fprime` to compute an approximated
      gradient.
    check_grad : compares the approximated gradient with the one provided
      by the model
    """

    def __init__(self, with_t1=False, with_t3=False, with_collisions=False):
        """Creates a quasistatic gradient descent solver with optional
        type1, type3 and collision detection and solving routines.

        Parameters
        ----------
        with_t1 : bool, default False
            whether or not to solve type 1 transitions at each
            iteration.
        with_t3 : bool, default False
            whether or not to solve type 3 transitions
            (i.e. elimnation of small triangular faces) at each
            iteration.
        with_collisions : bool, default False
            wheter or not to solve collisions

        Those corrections are applied in this order: first the type 1, then the
        type 3, then the collisions

        """
        self.set_pos = self._set_pos
        if with_t1:
            self.set_pos = auto_t1(self.set_pos)
        if with_t3:
            self.set_pos = auto_t3(self.set_pos)
        if with_collisions:
            self.set_pos = solve_collisions(self.set_pos)

    def find_energy_min(self, eptm, geom, model, **minimize_kw):
        """Energy minimization function.

        The epithelium's total energy is minimized by displacing its vertices.
        This is a wrapper around `scipy.optimize.minimize`

        Parameters
        ----------
        eptm : a :class:`tyssue.Epithlium` object
        geom : a geometry class
        geom must provide an `update_all` method that takes `eptm`
            as sole argument and updates the relevant geometrical quantities
        model : a model class
            model must provide `compute_energy` and `compute_gradient` methods
            that take `eptm` as first and unique positional argument.

        """

        settings = config.solvers.quasistatic()
        settings.update(**minimize_kw)
        pos0 = eptm.vert_df.loc[eptm.vert_df.is_active, eptm.coords].values.ravel()
        max_length = eptm.edge_df["length"].max() / 2
        bounds = np.vstack([pos0 - max_length, pos0 + max_length]).T
        res = optimize.minimize(
            self._opt_energy,
            pos0,
            args=(eptm, geom, model),
            bounds=bounds,
            jac=self._opt_grad,
            **settings
        )
        return res

    @staticmethod
    def _set_pos(eptm, geom, pos):
        ndims = len(eptm.coords)
        eptm.vert_df.loc[eptm.vert_df.is_active, eptm.coords] = pos.reshape(
            (pos.size // ndims, ndims)
        )
        geom.update_all(eptm)

    def _opt_energy(self, pos, eptm, geom, model):
        self.set_pos(eptm, geom, pos)
        return model.compute_energy(eptm)

    # The unused arguments bellow are legit, we need the same signature as _opt_energy
    @staticmethod
    def _opt_grad(pos, eptm, geom, model):
        grad_i = model.compute_gradient(eptm)

        return grad_i.loc[eptm.vert_df.is_active].values.ravel()

    @classmethod
    def approx_grad(cls, eptm, geom, model):
        pos0 = eptm.vert_df[eptm.coords].values.ravel()
        grad = optimize.approx_fprime(pos0, cls._opt_energy, 1e-9, eptm, geom, model)
        return grad

    @classmethod
    def check_grad(cls, eptm, geom, model):
        pos0 = eptm.vert_df[eptm.coords].values.ravel()
        grad_err = optimize.check_grad(
            cls._opt_energy, cls._opt_grad, pos0.flatten(), eptm, geom, model
        )
        return grad_err
