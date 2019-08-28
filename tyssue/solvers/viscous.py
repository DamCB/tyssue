"""Viscosity based ODE solver


"""
import logging
import numpy as np
import pandas as pd
import warnings

from itertools import count
from scipy.integrate import solve_ivp


from ..core.history import History
from ..behaviors.event_manager import EventManager
from ..behaviors.sheet.basic_events import reconnect


log = logging.getLogger(__name__)
MAX_ITER = 1000


def set_pos(eptm, geom, pos):
    """Updates the vertex position of the :class:`Epithelium` object.

    Assumes that pos is passed as a 1D array to be reshaped as (eptm.Nv, eptm.dim)

    """
    log.debug("set pos")
    eptm.vert_df.loc[eptm.active_verts, eptm.coords] = pos.reshape((-1, eptm.dim))
    geom.update_all(eptm)


class EulerSolver:
    """Explicit Euler solver



    """

    def __init__(
        self,
        eptm,
        geom,
        model,
        history=None,
        auto_reconnect=False,
        manager=None,
        bounds=None,
        with_t1=False,
        with_t3=False,
    ):
        """creates an instance of EulerSolver

        Parameters
        ----------
        eptm : a :class:`tyssue.Epithelium` instance
        geom : a Geometry class
        model : a Model class
        history : a :class:`tyssue.History` or :class:`tyssue.Hdf5History` instance
        auto_reconnect : bool
            if True, will automatically perform reconnections, default False
        manager : a :class:`tyssue.EventManager` instance
        bounds : tuple of (min, max),
            bonds the displacement of the vertices at each time step

        """

        self._set_pos = set_pos
        if with_t1:
            warnings.warn("with_t1 is deprecated and has no effect")
            # self._set_pos = auto_t1(self._set_pos)
        if with_t3:
            warnings.warn("with_t3 is deprecated and has no effect")
            # self._set_pos = auto_t3(self._set_pos)

        # self.rearange = with_t1 or with_t3
        # self.with_t3 = with_t3
        self.eptm = eptm
        self.geom = geom
        self.model = model
        if history is None:
            self.history = History(eptm)
        else:
            self.history = history
        self.prev_t = 0

        if auto_reconnect:
            if manager is None:
                manager = EventManager()
            if not "reconnect" in [n[0].__name__ for n in manager.next]:
                manager.append(reconnect)

        self.manager = manager
        self.bounds = bounds

    @property
    def current_pos(self):
        return self.eptm.vert_df[self.eptm.coords].values.ravel()

    def set_pos(self, pos):
        """Updates the eptm vertices position
        """
        return self._set_pos(self.eptm, self.geom, pos)

    def record(self, t):
        self.history.record(time_stamp=t)

    def solve(self, tf, dt, on_topo_change=None, topo_change_args=()):
        """Solves the system of differential equations from the current time
        to tf with steps of dt with a forward Euler method.

        Parameters
        ----------
        tf : float, final time when we stop solving
        dt : float, time step
        on_topo_change : function, optional, default None
             function of `self.eptm`
        topo_change_args : tuple, arguments passed to `on_topo_change`

        """
        for t in np.arange(self.prev_t, tf + dt, dt):
            pos = self.current_pos
            dot_r = self.ode_func(t, pos)
            if self.bounds is not None:
                dot_r = np.clip(dot_r, *self.bounds)
            pos = pos + dot_r * dt
            self.set_pos(pos)
            self.prev_t = t
            if self.manager is not None:
                self.manager.execute(self.eptm)
                self.geom.update_all(self.eptm)
                self.manager.update()

            if self.eptm.topo_changed:
                log.info("Topology changed")
                if on_topo_change is not None:
                    on_topo_change(*topo_change_args)
                self.eptm.topo_changed = False
            self.record(t)

    def ode_func(self, t, pos):
        """Computes the models' gradient.


        Returns
        -------
        dot_r : 1D np.ndarray of shape (self.eptm.Nv * self.eptm.dim, )

        .. math::
        \frac{dr_i}{dt} = \frac{\nabla U_i}{\eta_i}

        """

        grad_U = -self.model.compute_gradient(self.eptm)
        return (grad_U.values / self.eptm.vert_df["viscosity"].values[:, None]).ravel()


class IVPSolver:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            """It is not yet clear how to use scipy's `solve_ivp` with topology changes

Previous attempts where made but turned out to be clumsy..."""
        )
