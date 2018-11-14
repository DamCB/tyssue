"""
Isotropic energy model from Farhadifar et al. 2007 article
"""
import numpy as np
from ..geometry.sheet_geometry import SheetGeometry as sgeom

mu = 2 ** 1.5 * 3 ** 0.25


def elasticity(delta):
    return (delta ** 3 - 1) ** 2 / 2.0


def contractility(delta, gamma):
    return gamma * mu ** 2 * delta ** 2 / 2.0


def tension(delta, lbda):
    return lbda * mu * delta / 2.0


def isotropic_energies(sheet, model, geom, deltas, nondim_specs):

    # bck_face_shift = sheet.face_df['basal_shift']
    # bck_vert_shift = sheet.vert_df['basal_shift']
    # ## Faces only area and height

    V_0 = sheet.specs["face"]["prefered_vol"]
    vol_avg = sheet.face_df[sheet.face_df["is_alive"] == 1].vol.mean()
    rho_avg = sheet.vert_df.rho.mean()

    # ## Set height and volume to height0 and V0
    delta_0 = (V_0 / vol_avg) ** (1 / 3)
    geom.scale(sheet, delta_0, sheet.coords)

    h_0 = V_0 ** (1 / 3)

    sheet.face_df["basal_shift"] = rho_avg * delta_0 - h_0
    sheet.vert_df["basal_shift"] = rho_avg * delta_0 - h_0
    geom.update_all(sheet)

    energies = np.zeros((deltas.size, 3))
    for n, delta in enumerate(deltas):
        geom.scale(sheet, delta, sheet.coords + ["basal_shift"])
        geom.update_all(sheet)
        Et, Ec, Ev = model.compute_energy(sheet, full_output=True)
        energies[n, :] = [Et.sum(), Ec.sum(), Ev.sum()]
        geom.scale(sheet, 1 / delta, sheet.coords + ["basal_shift"])
        geom.update_all(sheet)
    energies /= sheet.face_df["is_alive"].sum()
    isotropic_relax(sheet, nondim_specs)

    return energies


def isotropic_relax(sheet, nondim_specs, geom=sgeom):
    """Deforms the sheet so that the faces pseudo-volume is at their
    isotropic optimum (on average)

    The specified model specs is assumed to be non-dimentional
    """

    vol0 = sheet.face_df["prefered_vol"].mean()
    h_0 = vol0 ** (1 / 3)
    live_faces = sheet.face_df[sheet.face_df.is_alive == 1]
    vol_avg = live_faces.vol.mean()
    rho_avg = sheet.vert_df.rho.mean()

    # ## Set height and volume to height0 and vol0
    delta = (vol0 / vol_avg) ** (1 / 3)
    geom.scale(sheet, delta, coords=sheet.coords)
    sheet.face_df["basal_shift"] = rho_avg * delta - h_0
    sheet.vert_df["basal_shift"] = rho_avg * delta - h_0
    geom.update_all(sheet)

    # ## Optimal value for delta
    delta_o = find_grad_roots(nondim_specs)
    if not np.isfinite(delta_o):
        raise ValueError("invalid parameters values")
    sheet.delta_o = delta_o
    # ## Scaling
    geom.scale(sheet, delta_o, coords=sheet.coords + ["basal_shift"])
    geom.update_all(sheet)


def isotropic_energy(delta, mod_specs):
    """
    Computes the theoritical energy per face for the given
    parameters.
    """
    lbda = mod_specs["edge"]["line_tension"]
    gamma = mod_specs["face"]["contractility"]
    elasticity_ = (delta ** 3 - 1) ** 2 / 2.0
    contractility_ = gamma * mu ** 2 * delta ** 2 / 2.0
    tension_ = lbda * mu * delta / 2.0
    energy = elasticity_ + contractility_ + tension_
    return energy


def isotropic_grad_poly(mod_specs):
    lbda = mod_specs["edge"]["line_tension"]
    gamma = mod_specs["face"]["contractility"]

    grad_poly = [3, 0, 0, -3, mu ** 2 * gamma, mu * lbda / 2.0]
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
