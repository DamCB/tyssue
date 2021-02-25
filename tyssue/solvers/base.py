import logging
from ..topology import TopologyChangeError

log = logging.getLogger(__name__)


def set_pos(eptm, geom, pos):
    """Updates the vertex position of the :class:`Epithelium` object.

    Assumes that pos is passed as a 1D array to be reshaped as (eptm.Nv, eptm.dim)

    """
    log.debug("set pos")
    if eptm.topo_changed:
        # reset the switch and interupt what we were doing
        eptm.topo_changed = False
        raise TopologyChangeError
    eptm.vert_df.loc[eptm.vert_df.is_active.astype(bool), eptm.coords] = pos.reshape(
        (-1, eptm.dim)
    )
    geom.update_all(eptm)
