'''
Isotropic functions
'''
import numpy as np

from ..utils.utils import update_default
from ..geometry.sheet_geometry import scale, update_all
from .sheet_vertex_model import get_default_mod_specs

mu = 6 * np.sqrt(2. / (3 * np.sqrt(3)))

def isotropic_relax(sheet, **mod_specs):
    """Deforms the sheet so that the cells area and
    pseudo-volume are at their isotropic optimum (on average)

    The specified model specs is assumed to be non-dimentional
    """
    mod_specs.update(get_default_mod_specs())

    area0 = mod_specs['cell']['prefered_area'][0]
    h_0 = mod_specs['cell']['prefered_height'][0]

    live_cells = sheet.cell_df[sheet.cell_df.is_alive==1]

    area_avg = live_cells.area.mean()
    rho_avg = live_cells.rho.mean()

    ### Set height and area to height0 and area0
    delta = (area0 / area_avg)**0.5
    scale(sheet, delta, coords=sheet.coords)
    sheet.cell_df['basal_shift'] = rho_avg * delta - h_0
    sheet.jv_df['basal_shift'] = rho_avg * delta - h_0
    update_all(sheet)

    ### Optimal value for delta
    delta_o = find_grad_roots(mod_specs)
    if not np.isfinite(delta_o):
        raise ValueError('invalid parameters values')
    sheet.delta_o = delta_o
    ### Scaling
    scale(sheet, delta_o, coords=sheet.coords+['basal_shift',])
    update_all(sheet)

def isotropic_energy(delta, mod_specs):
    """
    Computes the theoritical energy per cell for the given
    parameters.
    """
    lbda = mod_specs['je']['line_tension'][0]
    gamma = mod_specs['cell']['contractility'][0]
    elasticity = (delta**3 - 1 )**2 / 2.
    contractility = gamma * mu**2 * delta**2 / 2.
    tension = lbda * mu * delta / 2.
    energy = elasticity + contractility + tension
    return energy

def isotropic_grad_poly(mod_specs):
    lbda = mod_specs['je']['line_tension'][0]
    gamma = mod_specs['cell']['contractility'][0]
    grad_poly = [3, 0, 0,
                 -3,
                 mu**2 * gamma,
                 mu * lbda / 2.]
    return grad_poly

def isotropic_grad(mod_specs, delta):
    grad_poly = isotropic_grad_poly(mod_specs)
    return np.polyval(grad_poly, delta)

def find_grad_roots(mod_specs):
    poly = isotropic_grad_poly(mod_specs)
    roots = np.roots(poly)
    good_roots = np.real([r for r in roots if np.abs(r) == r])
    np.sort(good_roots)
    if len(good_roots) == 1:
        return good_roots
    elif len(good_roots) > 1:
        return good_roots[0]
    else:
        return np.nan
