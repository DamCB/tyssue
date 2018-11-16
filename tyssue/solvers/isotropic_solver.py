import numpy as np
import pandas as pd
import warnings
import logging

log = logging.getLogger(name=__name__)

from scipy import optimize

from ..utils import scaled_unscaled


def bruteforce_isotropic_relax(eptm, geom, model):
    def to_minimize(scale):
        return scaled_unscaled(model.compute_energy, scale, eptm, geom, args=[eptm])

    optim_res = optimize.minimize_scalar(
        to_minimize, method="bounded", bounds=[1e-6, 1e2]
    )
    if optim_res["success"]:
        log.info("Scaling by factor {:.3f}".format(optim_res["x"]))
        scale = optim_res["x"]
        geom.scale(eptm, scale, eptm.coords)
        geom.update_all(eptm)
    else:
        warnings.warn("Optimisation failed")

    return optim_res
