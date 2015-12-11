'''
Energy minimization solvers for the sheet vertex model
'''
from scipy import optimize

import numpy as np


def get_default_settings():
    default_settings = {
        'norm_factor': 1,
        'minimize': {
            'jac': opt_grad,
            'method': 'L-BFGS-B',
            'options': {'disp': False,
                        'ftol': 1e-6,
                        'gtol': 1e-3},
            }
        }
    return default_settings

def find_energy_min(sheet, geom, model,
                    pos_idx=None,
                    **settings_kw):

    settings = get_default_settings()
    settings.update(**settings_kw)

    coords = sheet.coords
    if pos_idx is None:
        pos0 = sheet.jv_df[coords].values.ravel()
        pos_idx = sheet.jv_df.index
    else:
        pos0 = sheet.jv_df.loc[pos_idx, coords].values.ravel()

    max_length = 2 * sheet.je_df['length'].max()
    bounds = np.vstack([pos0 - max_length,
                        pos0 + max_length]).T
    if settings['minimize']['jac'] is None:
        return
    res = optimize.minimize(opt_energy, pos0,
                            args=(pos_idx, sheet, geom, model),
                            bounds=bounds, **settings['minimize'])
    return res

def set_pos(pos, pos_idx, sheet):
    ndims = len(sheet.coords)
    pos_ = pos.reshape((pos.size//ndims, ndims))
    sheet.jv_df.loc[pos_idx, sheet.coords] = pos_

def opt_energy(pos, pos_idx, sheet, geom, model):
    set_pos(pos, pos_idx, sheet)
    geom.update_all(sheet)
    return model.compute_energy(sheet, full_output=False)

# The unused arguments bellow are legit, need same call sig as above
def opt_grad(pos, pos_idx, sheet, geom, model):
    grad_i = model.compute_gradient(sheet, components=False)
    return grad_i.values.flatten()

def approx_grad(sheet, geom, model):
    pos0 = sheet.jv_df[sheet.coords].values.ravel()
    pos_idx = sheet.jv_idx
    grad = optimize.approx_fprime(pos0,
                                  opt_energy,
                                  1e-9, pos_idx,
                                  sheet, geom, model)
    return grad


def check_grad(sheet, geom, model):

    pos0 = sheet.jv_df[sheet.coords].values.ravel()
    pos_idx = sheet.jv_idx
    grad_err = optimize.check_grad(opt_energy,
                                   opt_grad,
                                   pos0.flatten(),
                                   pos_idx,
                                   sheet, geom, model)
    return grad_err
