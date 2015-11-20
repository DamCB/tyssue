'''
Energy minimization solvers for the sheet vertex model
'''
from scipy import optimize

from ..geometry import sheet_geometry as geom
from ..dynamics import sheet_vertex_model as model
import numpy as np


def get_default_settings():
    default_settings = {
        'norm_factor': 1,
        'minimize':{
            'jac': None,
            'method':'L-BFGS-B',
            'options': {'disp':False,
                        'gtol':1e-3},
            }
        }
    return default_settings

def find_energy_min(sheet, pos_idx=None,
                    coords=None, **settings_kw):

    settings = get_default_settings()
    settings.update(**settings_kw)

    if coords is None:
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
                            args=(pos_idx, sheet, coords),
                            bounds=bounds, **settings['minimize'])
    return res

def set_pos(pos, pos_idx, sheet, coords):
    pos_ = pos.reshape((pos.size//3, 3))
    sheet.jv_df.loc[pos_idx, coords] = pos_

def opt_energy(pos, pos_idx, sheet, coords):
    set_pos(pos, pos_idx, sheet, coords)
    geom.update_all(sheet)
    return model.compute_energy(sheet, full_output=False)

# The unused arguments bellow are legit, need same call sig as above
def opt_grad(pos, pos_idx, sheet, coords):
    grad_i = model.compute_gradient(sheet, components=False)
    return grad_i.values.flatten()

def approx_grad(sheet, coords):
    pos0 = sheet.jv_df[coords].values.ravel()
    pos_idx = sheet.jv_idx
    grad = optimize.approx_fprime(pos0,
                                  opt_energy,
                                  1e-9, pos_idx,
                                  sheet, coords)
    return grad


def check_grad(sheet, coords):

    pos0 = sheet.jv_df[coords].values.ravel()
    pos_idx = sheet.jv_idx
    grad_err = optimize.check_grad(opt_energy,
                                   opt_grad,
                                   pos0.flatten(),
                                   pos_idx,
                                   sheet, coords)
    return grad_err
