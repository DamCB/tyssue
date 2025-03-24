"""
Energy minimization solvers for the sheet vertex model
"""
from scipy import optimize
from .. import config
import numpy as np


class Solver:
    @classmethod
    def find_energy_min(cls, sheet, geom, model, pos_idx=None, **settings_kw):
        # Loads 'tyssue/config/solvers/minimize.json
        settings = config.solvers.minimize_spec()
        settings.update(**settings_kw)

        coords = sheet.coords
        if pos_idx is None:
            pos_idx = sheet.vert_df[sheet.vert_df["is_active"] == 1].index

        pos0 = sheet.vert_df.loc[pos_idx, coords].values.ravel()

        max_length = 2 * sheet.edge_df["length"].max()
        bounds = np.vstack([pos0 - max_length, pos0 + max_length]).T
        res = optimize.minimize(
            cls.opt_energy,
            pos0,
            args=(pos_idx, sheet, geom, model),
            bounds=bounds,
            jac=cls.opt_grad,
            **settings["minimize"]
        )
        return res

    @staticmethod
    def set_pos(pos, pos_idx, sheet):
        ndims = len(sheet.coords)
        pos_ = pos.reshape((pos.size // ndims, ndims))
        sheet.vert_df.loc[pos_idx, sheet.coords] = pos_

    @classmethod
    def opt_energy(cls, pos, pos_idx, sheet, geom, model):
        cls.set_pos(pos, pos_idx, sheet)
        geom.update_all(sheet)
        return model.compute_energy(sheet, full_output=False)

    # The unused arguments bellow are legit, need same call sig as above
    @staticmethod
    def opt_grad(pos, pos_idx, sheet, geom, model):
        grad_i = model.compute_gradient(sheet, components=False)
        grad_i = grad_i.loc[pos_idx]
        return grad_i.values.flatten()

    @classmethod
    def approx_grad(cls, sheet, geom, model):
        pos0 = sheet.vert_df[sheet.coords].values.ravel()
        pos_idx = sheet.vert_df[sheet.vert_df["is_active"] == 1].index

        grad = optimize.approx_fprime(
            pos0, cls.opt_energy, 1e-9, pos_idx, sheet, geom, model
        )
        return grad

    @classmethod
    def check_grad(cls, sheet, geom, model):

        pos_idx = sheet.vert_df[sheet.vert_df["is_active"] == 1].index
        pos0 = sheet.vert_df.loc[pos_idx, sheet.coords].values.ravel()

        grad_err = optimize.check_grad(
            cls.opt_energy, cls.opt_grad, pos0.flatten(), pos_idx, sheet, geom, model
        )
        return grad_err
