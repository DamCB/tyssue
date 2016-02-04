'''
Isotropic functions
'''
import numpy as np

from ..geometry.base_geometry import scale
from ..geometry.sheet_geometry import update_all

mu = 6 * np.sqrt(2. / (3 * np.sqrt(3)))

def elasticity(delta):
    return (delta**3 - 1 )**2 / 2.

def contractility(delta, gamma):
    return gamma * mu**2 * delta**2 / 2.

def tension(delta, lbda):
    return lbda * mu * delta / 2.

def isotropic_energies(sheet, model, geom,
                       deltas, nondim_specs):

    ### Faces only area and height
    area_avg = sheet.face_df[sheet.face_df['is_alive'] == 1].area.mean()
    rho_avg = sheet.jv_df.rho.mean()
    area0 = sheet.specs['face']['prefered_area']
    h_0 = sheet.specs['face']['prefered_height']

    ### Set height and area to height0 and area0
    delta_0 = (area0 / area_avg)**0.5
    geom.scale(sheet, delta_0, sheet.coords)
    sheet.face_df['basal_height'] = rho_avg * delta_0 - h_0
    geom.update_all(sheet)

    energies = np.zeros((deltas.size, 3))
    for n, delta in enumerate(deltas):
        geom.scale(sheet, delta,
                   sheet.coords+['basal_shift'])
        geom.update_all(sheet)

        Et, Ec, Ev = model.compute_energy(sheet, full_output=True)
        energies[n, :] = [Et.sum(), Ec.sum(), Ev.sum()]
        geom.scale(sheet, 1/delta, sheet.coords+['basal_shift'])
        geom.update_all(sheet)

    energies /= sheet.face_df['is_alive'].sum()
    isotropic_relax(sheet, nondim_specs)
    return energies


def isotropic_relax(sheet, nondim_specs):
    """Deforms the sheet so that the faces area and
    pseudo-volume are at their isotropic optimum (on average)

    The specified model specs is assumed to be non-dimentional
    """

    area0 = sheet.face_df['prefered_area'].mean()
    h_0 = sheet.face_df['prefered_height'].mean()

    live_faces = sheet.face_df[sheet.face_df.is_alive==1]

    area_avg = live_faces.area.mean()
    rho_avg = sheet.jv_df.rho.mean()

    ### Set height and area to height0 and area0
    delta = (area0 / area_avg)**0.5
    scale(sheet, delta, coords=sheet.coords)
    sheet.face_df['basal_shift'] = rho_avg * delta - h_0
    sheet.jv_df['basal_shift'] = rho_avg * delta - h_0
    update_all(sheet)

    ### Optimal value for delta
    delta_o = find_grad_roots(nondim_specs)
    if not np.isfinite(delta_o):
        raise ValueError('invalid parameters values')
    sheet.delta_o = delta_o
    ### Scaling
    scale(sheet, delta_o, coords=sheet.coords+['basal_shift',])
    update_all(sheet)

def isotropic_energy(delta, mod_specs):
    """
    Computes the theoritical energy per face for the given
    parameters.
    """
    lbda = mod_specs['je']['line_tension']
    gamma = mod_specs['face']['contractility']
    elasticity_ = (delta**3 - 1 )**2 / 2.
    contractility_ = gamma * mu**2 * delta**2 / 2.
    tension_ = lbda * mu * delta / 2.
    energy = elasticity_ + contractility_ + tension_
    return energy

def isotropic_grad_poly(mod_specs):
    lbda = mod_specs['je']['line_tension']
    gamma = mod_specs['face']['contractility']
    grad_poly = [3, 0, 0, -3,
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
        print('multiple roots')
        return good_roots[0]
    else:
        return np.nan
