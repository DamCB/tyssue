"""Viscosity based ODE solver


"""
import logging
import numpy as np
from itertools import count
from scipy.integrate import solve_ivp

from ..collisions import auto_collisions
from ..topology import auto_t1, auto_t3
from ..core.history import History
from .base import set_pos
from .base import TopologyChangeError


log = logging.getLogger(__name__)

MAX_ITER = 10


class ViscousSolver:
    def __init__(
        self,
        eptm,
        geom,
        model,
        history=None,
        with_collisions=False,
        with_t1=False,
        with_t3=False,
    ):

        self._set_pos = set_pos
        if with_collisions:
            self._set_pos = auto_collisions(self._set_pos)
        if with_t1:
            self._set_pos = auto_t1(self._set_pos)
        if with_t3:
            self._set_pos = auto_t3(self._set_pos)
        self.rearange = with_t1 or with_t3
        self.eptm = eptm
        self.geom = geom
        self.model = model
        if history is None:
            self.history = History(eptm)
        else:
            self.history = history
        self.prev_t = 0
        self.solver_t = np.empty((0,))

    def set_pos(self, pos):
        return self._set_pos(self.eptm, self.geom, pos)

    def ode_func(self, t, pos):
        self.set_pos(pos)
        # self.history.record(to_record=["vert", "edge", "face", "cell"], time_stamp=t)
        grad_U = self.model.compute_gradient(self.eptm)
        return (grad_U.values / self.eptm.vert_df["viscosity"].values[:, None]).ravel()

    def solve(self, t):
        res = {"message": "Not started", "success": False}
        for i in count():
            if i == MAX_ITER:
                res["message"] = res["message"] + "\nMax number of iterations reached!"
                return res
            try:
                _prev_t = self.prev_t
                self.prev_t = t
                pos0 = self.eptm.vert_df[self.eptm.coords].values.ravel()
                res = solve_ivp(self.ode_func, (_prev_t, t), pos0)
                self.solver_t = np.concatenate((self.solver_t, res["t"]))
                return res
            except TopologyChangeError:
                print(f"Topology changed at time {t}")
