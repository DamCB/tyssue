import logging

log = logging.getLogger(__name__)


class TopologyChangeError(ValueError):
    """ Raised when trying to assign values without
    the correct length to an epithelium dataset
    """

    pass


def set_pos(eptm, geom, pos):
    """Updates the vertex position of the :class:`Epithelium` object.

    Assumes that pos is passed as a 1D array to be reshaped as (eptm.Nv, eptm.dim)

    """
    log.debug("set pos")
    if eptm.topo_changed:
        # reset the switch and interupt what we were doing
        eptm.topo_changed = False
        raise TopologyChangeError
    eptm.vert_df.loc[eptm.active_verts, eptm.coords] = pos.reshape((-1, eptm.dim))
    geom.update_all(eptm)
