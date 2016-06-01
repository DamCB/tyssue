import numpy as np
from scipy import optimize

from tyssue.solvers.sheet_vertex_solver import Solver

from ..config.solvers import minimize_spec


class MultiSheetSolver(Solver):

    @classmethod
    def find_energy_min(cls, msheet, geom,
                        model, pos_idxs, **settings_kw):
        settings = minimize_spec()
        settings.update(**settings_kw)

        pos0 = [sheet.vert_df.loc[pos_idx, sheet.coords].values.flatten()
                for sheet, pos_idx in zip(msheet, pos_idxs)]

        max_length = 2 * msheet[0].edge_df['length'].max()
        pos0 = np.concatenate(pos0)
        bounds = np.vstack([pos0 - max_length,
                            pos0 + max_length]).T

        res = optimize.minimize(cls.opt_energy, pos0,
                                args=(pos_idxs, msheet, geom, model),
                                jac=cls.opt_grad, bounds=bounds,
                                **settings['minimize'])
        return res

    @staticmethod
    def set_pos(pos, pos_idxs, msheet):
        v_idxs = np.array([pos_idx.shape[0] for
                           pos_idx in pos_idxs]).cumsum()
        for sheet, pos_chunk, pos_idx in zip(msheet,
                                             np.split(pos, v_idxs[:-1]*3),
                                             pos_idxs):
            sheet.vert_df.loc[pos_idx, sheet.coords] = pos_chunk.reshape(
                (pos_chunk.size//3, 3))

    @staticmethod
    def opt_grad(pos, pos_idxs, msheet, geom, model):
        grads = model.compute_gradient(msheet, components=False)
        grad_i = np.concatenate([grad.loc[pos_idx].values.flatten()
                                 for grad, pos_idx
                                 in zip(grads, pos_idxs)])
        return grad_i

    @classmethod
    def check_grad(cls, msheet, geom, model):

        pos_idxs = [sheet.vert_df[sheet.vert_df['is_active'] == 1].index
                    for sheet in msheet]
        pos0 = [sheet.vert_df.loc[pos_idx, sheet.coords].values.flatten()
                for sheet, pos_idx in zip(msheet, pos_idxs)]

        pos0 = np.concatenate(pos0)

        grad_err = optimize.check_grad(cls.opt_energy,
                                       cls.opt_grad,
                                       pos0,
                                       pos_idxs,
                                       msheet, geom, model)
        return grad_err
