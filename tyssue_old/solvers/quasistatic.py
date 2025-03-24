"""Quasistatic solver for vertex models

"""
import numpy as np
import logging
from itertools import count

from scipy import optimize
from .. import config
from ..collisions import auto_collisions
from ..topology import auto_t1, auto_t3

from .base import TopologyChangeError, set_pos

log = logging.getLogger(__name__)

MAX_ITER = 100


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

    def __init__(self, with_collisions=False, with_t1=False, with_t3=False):
        """Creates a quasistatic gradient descent solver with optional
        type1, type3 and collision detection and solving routines.

        Parameters
        ----------
        with_collisions : bool, default False
            wheter or not to solve collisions
        with_t1 : bool, default False
            whether or not to solve type 1 transitions at each
            iteration.
        with_t3 : bool, default False
            whether or not to solve type 3 transitions
            (i.e. elimnation of small triangular faces) at each
            iteration.

        Those corrections are applied in this order: first the type 1, then the
        type 3, then the collisions

        """
        self.set_pos = set_pos
        if with_t1:
            self.set_pos = auto_t1(self.set_pos)
        if with_t3:
            self.set_pos = auto_t3(self.set_pos)
        if with_collisions:
            self.set_pos = auto_collisions(self.set_pos)
        self.restart = True
        self.rearange = with_t1 or with_t3
        self.res = {"success": False, "message": "Not Started"}
        self.num_restarts = 0

    def find_energy_min(self, eptm, geom, model, periodic=False, **minimize_kw):
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
        log.info("initial number of vertices: %i", eptm.Nv)
        settings = config.solvers.quasistatic()
        settings.update(**minimize_kw)
        if periodic == False:
            res = self._minimize(eptm, geom, model, **settings)
        else:
            res = self._minimize_pbc(eptm, geom, model, **settings)
        log.info("final number of vertices: %i", eptm.Nv)

        return res

    def _minimize(self, eptm, geom, model, **kwargs):
        for i in count():
            if i == MAX_ITER:
                return self.res
            pos0 = eptm.vert_df.loc[
                eptm.vert_df.is_active.astype(bool), eptm.coords
            ].values.flatten()
            try:
                self.res = optimize.minimize(
                    self._opt_energy,
                    pos0,
                    args=(eptm, geom, model),
                    jac=self._opt_grad,
                    **kwargs
                )
                return self.res
            except TopologyChangeError:
                log.info("TopologyChange")
                self.num_restarts = i + 1

    def _opt_energy(self, pos, eptm, geom, model):
        if self.rearange and eptm.topo_changed:
            # reset switch
            eptm.topo_changed = False
            raise TopologyChangeError("Topology changed before energy evaluation")
        self.set_pos(eptm, geom, pos)
        return model.compute_energy(eptm)

    # The unused arguments bellow are legit, we need the same signature as _opt_energy
    def _opt_grad(self, pos, eptm, geom, model):
        if self.rearange and eptm.topo_changed:
            raise TopologyChangeError("Topology changed before gradient evaluation")
        grad_i = model.compute_gradient(eptm)
        return grad_i.loc[eptm.vert_df.is_active.astype(bool)].values.ravel()

    def approx_grad(self, eptm, geom, model):
        pos0 = eptm.vert_df.loc[
            eptm.vert_df.is_active.astype(bool), eptm.coords
        ].values.ravel()
        grad = optimize.approx_fprime(pos0, self._opt_energy, 1e-9, eptm, geom, model)
        return grad

    def check_grad(self, eptm, geom, model):
        pos0 = eptm.vert_df.loc[
            eptm.vert_df.is_active.astype(bool), eptm.coords
        ].values.ravel()
        grad_err = optimize.check_grad(
            self._opt_energy, self._opt_grad, pos0.flatten(), eptm, geom, model
        )
        return grad_err

    # The functions bellow are for a perodic square tissue in 2D
    def _minimize_pbc(self, eptm, geom, model, **kwargs):
        for i in count():
            if i == MAX_ITER:
                return self.res
            pos0 = eptm.vert_df.loc[
                eptm.vert_df.is_active.astype(bool), eptm.coords
            ].values.flatten()
            for u in eptm.settings["boundaries"]:
                size = eptm.specs["settings"]["boundaries"][u][1]
            pos0 = np.append(pos0, size)
            try:
                self.res = optimize.minimize(
                    self._opt_energy_pbc,
                    pos0,
                    args=(eptm, geom, model),
                    jac=self._opt_grad_pbc,
                    **kwargs
                )
                return self.res
            except TopologyChangeError:
                log.info("TopologyChange")
                self.num_restarts = i + 1

    def _opt_energy_pbc(self, pos, eptm, geom, model):
        if self.rearange and eptm.topo_changed:
            # reset switch
            eptm.topo_changed = False
            raise TopologyChangeError("Topology changed before energy evaluation")
        for u, boundary in eptm.settings["boundaries"].items():
            eptm.specs["settings"]["boundaries"][u] = [boundary[0], pos[-1]]
        self.set_pos(eptm, geom, pos[0:-1])
        geom.update_all(eptm)
        e = model.compute_energy(eptm)
        return e

    def _opt_grad_pbc(self, pos, eptm, geom, model, box_increment=0.0001):
        """Gradient calculation for vertices along with an approximation for box size variable

        Paramameters
        ------------
        pos: positions of vertices and last element is size of periodic box
        eptm: a :class:`tyssue.Epithlium` object ( function was made for 2D arrangments)
        geom : a geometry class
        geom must provide an `update_all` method that takes `eptm`
            as sole argument and updates the relevant geometrical quantities
        model : a model class
            model must provide `compute_energy` and `compute_gradient` methods
            that take `eptm` as first and unique positional argument.
        box_increment: size of displacement of box size to approximate gradient of box size variable

        """
        if self.rearange and eptm.topo_changed:
            raise TopologyChangeError("Topology changed before gradient evaluation")
        grad_i = model.compute_gradient(eptm)
        energy_before_increment = model.compute_energy(eptm)
        for u, boundary in eptm.settings["boundaries"].items():
            eptm.specs["settings"]["boundaries"][u] = [
                boundary[0],
                boundary[1] + box_increment,
            ]
        geom.update_all(eptm)
        energy_after_increment = model.compute_energy(eptm)
        grad_of_box_size_variable = (
            energy_after_increment - energy_before_increment
        ) / box_increment
        for u, boundary in eptm.settings["boundaries"].items():
            eptm.specs["settings"]["boundaries"][u] = [
                boundary[0],
                boundary[1] - box_increment,
            ]
        return np.append(
            grad_i.loc[eptm.vert_df.is_active.astype(bool)].values.ravel(),
            grad_of_box_size_variable,
        )
