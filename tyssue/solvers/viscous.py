"""Viscosity based ODE solver


"""
import logging
import numpy as np
import pandas as pd

from itertools import count

from scipy.integrate import solve_ivp

from ..collisions import auto_collisions
from ..topology import all_rearangements, auto_t1, auto_t3

from ..core.history import History
from ..core.sheet import Sheet
from .base import set_pos, TopologyChangeError


log = logging.getLogger(__name__)
MAX_ITER = 1000


class EulerSolver:
    """
    """

    def __init__(
        self,
        eptm,
        geom,
        model,
        history=None,
        with_t1=False,
        with_t3=False,
        manager=None,
    ):
        self._set_pos = set_pos
        if with_t1:
            self._set_pos = auto_t1(self._set_pos)
        if with_t3:
            self._set_pos = auto_t3(self._set_pos)

        self.rearange = with_t1 or with_t3
        self.with_t3 = with_t3
        self.eptm = eptm
        self.geom = geom
        self.model = model
        if history is None:
            self.history = History(eptm)
        else:
            self.history = history
        self.prev_t = 0
        self.manager = manager

    @property
    def pos0(self):
        return self.eptm.vert_df[self.eptm.coords].values.ravel()

    def set_pos(self, pos):
        """Updates the eptm vertices position
        """
        return self._set_pos(self.eptm, self.geom, pos)

    def record(self, t):
        self.history.record(["vert"], t)

    def solve(self, tf, dt, **solver_kwargs):

        pos = self.pos0
        for t in np.arange(self.prev_t, tf + dt, dt):
            try:
                dot_r = self.ode_func(t, pos)
                pos = pos + dot_r * dt

            except TopologyChangeError:
                log.info("Topology changed")
                self.history.record(["face", "edge"], t)
                if "cell" in self.eptm.datasets:
                    self.history.record(["cell"], t)
                pos = self.pos0
                self.geom.update_all(self.eptm)

            self.record(t)

    def ode_func(self, t, pos):
        """Updates the vertices positions and computes the gradient.

        Returns
        -------
        dot_r : 1D np.ndarray of shape (self.eptm.Nv * self.eptm.dim, )

        .. math::
        \frac{dr_i}{dt} = \frac{\nabla U_i}{\heta_i}

        """
        self.set_pos(pos)
        if self.eptm.topo_changed:
            self.eptm.topo_changed = False
            raise TopologyChangeError
        self.prev_t = t
        if self.manager is not None:
            self.manager.execute(self.eptm)

        grad_U = -self.model.compute_gradient(self.eptm)

        if self.manager is not None:
            self.manager.update()

        return (grad_U.values / self.eptm.vert_df["viscosity"].values[:, None]).ravel()


class IVPSolver(EulerSolver):
    """
    """

    def solve(self, t, **solver_kwargs):
        """
        """
        res = {"message": "Not started", "success": False}
        for i in count():
            if i == MAX_ITER:
                res["message"] = res["message"] + "\nMax number of iterations reached!"
                return res
            pos0 = self.eptm.vert_df[self.eptm.coords].values.ravel()
            if self.prev_t > t:
                return
            if "t_eval" in solver_kwargs:
                solver_kwargs["t_eval"] = solver_kwargs["t_eval"][
                    np.where(solver_kwargs["t_eval"] >= self.prev_t)[0]
                ]

            try:
                res = solve_ivp(self.ode_func, (self.prev_t, t), pos0, **solver_kwargs)
            except TopologyChangeError:
                log.info("Topology changed")
                self.history.record(["face", "edge"], t)
                if "cell" in self.eptm.datasets:
                    self.history.record(["cell"], t)
                continue

            self.record(res)
            return res

    def record(self, res):
        """Records the solution
        """
        positions = res.y.T.reshape((-1, 3))

        vert_id = np.tile(self.eptm.vert_df.index, res.t.shape[0])

        times = np.repeat(res.t, self.eptm.Nv)
        hist = pd.DataFrame(
            index=np.arange(times.shape[0]), columns=["vert", *self.eptm.coords, "time"]
        )
        hist["vert"] = vert_id
        hist["time"] = times
        hist[self.eptm.coords] = positions

        self.history.datasets["vert"] = pd.concat(
            (self.history.datasets["vert"], hist), ignore_index=True, sort=False
        )
        self.history.record(["face", "edge"], res.t[-1])
        if "cell" in self.eptm.datasets:
            self.history.record(["cell"], res.t[-1])
