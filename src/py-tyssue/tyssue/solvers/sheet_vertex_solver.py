'''
Energy minimization solvers for the sheet vertex model
'''
from scipy import optimize

from ..geometry import sheet_geometry as geom
from ..dynamics import sheet_vertex_model as model

from ..utils.utils import (_to_3d, set_data_columns,
                           update_default)

default_params = {
    'l_bfgs_b_options': {'disp':False,
                         'gtol':1e-3},
    'min_length':1e-6,
    }

def find_energy_min(sheet, pos_idx=None, 
                    coords=None, parameters=None):

    parameters = update_default(default_params, parameters)
    if coords is None:
        coords = sheet.coords
    if pos_idx is None:
        pos0 = sheet.jv_df[coords].values.ravel()
    else:
        pos0 = sheet.jv_df.loc[pos_idx, coords].values.ravel()
    
    max_length = sheet.je_df['length'].max()
    min_length = parameters['min_length']
    bounds = np.vstack([pos0 - min_length, 
                        pos0 + max_length]).T

    l_bfgs_b_options = default_params['l_bfgs_b_options']
    res = optimize.minimize(opt_energy, pos0,
                            args=(sheet, coords),
                            bounds=bounds,
                            jac=opt_grad,
                            method='L-BFGS-B',
                            options=l_bfgs_b_options)
    return res

def set_pos(pos, pos_idx, sheet, coords):
    pos = pos.reshape((pos.size//3, 3))
    sheet.jv_df.loc[pos_idx, coords] = pos

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
