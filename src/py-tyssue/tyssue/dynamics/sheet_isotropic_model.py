'''
Isotropic functions
'''
import numpy as np

from ..utils.utils import update_default
from ..geometry.sheet_geometry import scale, update_all
from .sheet_vertex_model import default_params

mu = 6 * np.sqrt(2. / (3 * np.sqrt(3)))

def isotropic_relax(sheet, adim_parameters=None):

    adim_parameters = update_default(default_params, 
                                    adim_parameters)
    area0 = adim_parameters['prefered_area']
    h_0 = adim_parameters['prefered_height']

    live_cell_idx = (sheet.cell_df.is_alive==1).index
    live_cells = sheet.cell_df.loc[live_cell_idx]
    
    area_avg = live_cells.area.mean()
    rho_avg = live_cells.rho.mean()

    ### Set height and area to height0 and area0
    delta = (area0 / area_avg)**0.5
    scale(sheet, delta, coords=sheet.coords)
    sheet.cell_df['rho_lumen'] = rho_avg * delta - h_0
    sheet.jv_df['rho_lumen'] = rho_avg * delta - h_0
    update_all(sheet)
    
    ### Optimal value for delta
    delta_o = find_grad_roots(adim_parameters)
    if not np.isfinite(delta_o):
        raise ValueError('invalid parameters values')
    sheet.delta_o = delta_o
    ### Scaling
    scale(sheet, delta_o, coords=sheet.coords+['rho_lumen',])
    update_all(sheet)

def isotropic_energy(delta, adim_parameters):
    """
    Computes the theoritical energy per cell for the given
    parameters.
    """
    adim_parameters = update_default(default_params, 
                                     adim_parameters)
    lbda = adim_parameters['line_tension']
    gamma = adim_parameters['contractility']

    elasticity = (delta**3 - 1 )**2 / 2.
    contractility = gamma * mu**2 * delta**2 / 2.
    tension = lbda * mu * delta / 2.
    energy = elasticity + contractility + tension
    return energy

def isotropic_grad_poly(adim_parameters):
    lbda = adim_parameters['line_tension']
    gamma = adim_parameters['contractility']
    grad_poly = [3, 0, 0,
                 -3,
                 mu**2 * gamma,
                 mu * lbda / 2.]
    return grad_poly

def isotropic_grad(adim_parameters, delta):
    adim_parameters = update_default(default_params, 
                                     adim_parameters)
    grad_poly = isotropic_grad_poly(adim_parameters)
    return np.polyval(grad_poly, delta)

def find_grad_roots(adim_parameters):
    adim_parameters = update_default(default_params, 
                                     adim_parameters)
    p = isotropic_grad_poly(adim_parameters)
    roots = np.roots(p)
    good_roots = np.real([r for r in roots if np.abs(r) == r])
    np.sort(good_roots)
    if len(good_roots) == 1:
        return good_roots
    elif len(good_roots) > 1:
        return good_roots[0]
    else:
        return np.nan

