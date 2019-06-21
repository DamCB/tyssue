"""Viscosity based ODE solver


"""
import logging
import numpy as np
import pandas as pd

from itertools import count

from scipy.integrate import solve_ivp

from ..collisions import auto_collisions
from ..topology import single_rearangement
from ..core.history import History
from ..core.sheet import Sheet
from .base import set_pos


log = logging.getLogger(__name__)

MAX_ITER = 10


class ViscousSolver:
    """
    """

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

    def set_pos(self, pos):
        """Updates the eptm vertices position
        """
        return self._set_pos(self.eptm, self.geom, pos)

    def ode_func(self, t, pos):
        """Updates the vertices positions and computes the gradient.

        Returns
        -------
        les : np.ndarray of shape (self.eptm.Nv * self.eptm.dim, )

        .. math::
        \frac{\nabla U_i}{\heta_i}

        """
        self.set_pos(pos)
        self.prev_t = t
        # self.history.record(to_record=["vert", "edge", "face", "cell"], time_stamp=t)
        grad_U = -self.model.compute_gradient(self.eptm)
        return (grad_U.values / self.eptm.vert_df["viscosity"].values[:, None]).ravel()

    def solve(self, t, t0=0.0, **solver_kwargs):
        """Finds a solution to the intial value problem

        .. math ::
        \heta_i \dot{\mathbf{r}_{i}} = \grad_i U


        """
        self.prev_t = t0

        if self.rearange:
            event = _short_event(self.eptm, self.geom)
            event.terminal = True
            event.direction = -1
            solver_kwargs["events"] = event

        print(solver_kwargs)

        res = {"message": "Not started", "success": False}
        for i in count():
            print(i)
            if i == MAX_ITER:
                res["message"] = res["message"] + "\nMax number of iterations reached!"
                return res
            pos0 = self.eptm.vert_df[self.eptm.coords].values.ravel()
            res = solve_ivp(self.ode_func, (self.prev_t, t), pos0, **solver_kwargs)
            self.record(res)
            if res.t_events is None:
                return res
            elif res.t_events[0].shape[0] == 0:
                continue
            else:
                log.info("Rearanging")
                single_rearangement(self.eptm, with_t3=self.with_t3)

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


def _short_event(eptm, geom):
    def short_func(t, pos):
        log.info("evaluating short function")
        set_pos(eptm, geom, pos)
        threshold_length = eptm.settings.get("threshold_length", 1e-4)
        min_length = eptm.edge_df.length.min()
        return min_length - threshold_length * 0.9

    return short_func
