"""Quasistatic solver for vertex models

"""
import logging
from itertools import count

import numpy as np
from scipy import optimize

from .. import config
from ..topology import _t2_flag, auto_t1, auto_t2
from .base import TopologyChangeError, set_pos

log = logging.getLogger(__name__)

MAX_ITER = 100
# Smallest allowed period of a periodic box
MIN_PERIOD = 1e-2


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

    def __init__(self, with_t1=False, with_t2=False, with_t3=None):
        """Creates a quasistatic gradient descent solver with optional
        type1 and type2 transition solving routines.

        Parameters
        ----------
        with_t1 : bool, default False
            whether or not to solve type 1 transitions at each
            iteration.
        with_t2 : bool, default False
            whether or not to solve type 2 transitions
            (i.e. elimnation of small triangular faces) at each
            iteration.
        with_t3 : bool, optional
            deprecated alias of `with_t2`

        Those corrections are applied in this order: first the type 1, then the
        type 2

        """
        with_t2 = _t2_flag(with_t2, with_t3)
        self.set_pos = set_pos
        if with_t1:
            self.set_pos = auto_t1(self.set_pos)
        if with_t2:
            self.set_pos = auto_t2(self.set_pos)
        self.restart = True
        self.rearange = with_t1 or with_t2
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
        if not periodic:
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
            # the last degree of freedom is the box's upper bound, which must
            # stay above the lower bounds
            lowest = max(
                boundary[0] for boundary in eptm.settings["boundaries"].values()
            )
            bounds = [(None, None)] * (pos0.size - 1) + [(lowest + MIN_PERIOD, None)]
            try:
                self.res = optimize.minimize(
                    self._opt_energy_pbc,
                    pos0,
                    args=(eptm, geom, model),
                    jac=self._opt_grad_pbc,
                    bounds=bounds,
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
        box_increment: size of displacement of box size to approximate gradient
            of box size variable

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
