"""
Specific functions for apoptosis vertex model
"""

from ..utils.utils import _to_3d
from .sheet_vertex_model import SheetModel, AnchoredSheetModel
from .sheet_gradients import height_grad


class SheetApoptosisModel(SheetModel):

    @staticmethod
    def compute_energy(sheet, full_output=False):
        nrj_norm_factor = sheet.specs['settings']['nrj_norm_factor']

        E_r = sheet.vert_df.eval('height * radial_tension * is_active')
        if full_output:
            energies = SheetModel.compute_energy(sheet,
                                                 full_output=True)
            return energies + (E_r / nrj_norm_factor,)
        else:
            E_base = SheetModel.compute_energy(sheet, full_output=False)
            return E_base + E_r.sum() / nrj_norm_factor

    @staticmethod
    def compute_gradient(sheet, components=False):

        nrj_norm_factor = sheet.specs['settings']['nrj_norm_factor']
        base_grad = SheetModel.compute_gradient(sheet,
                                                components=False)

        rad_grad = (_to_3d(sheet.vert_df['radial_tension']) *
                    height_grad(sheet)) / nrj_norm_factor
        if components:
            return base_grad, rad_grad

        return base_grad + rad_grad


class AnchoredSheetApoptosisModel(AnchoredSheetModel):

    @staticmethod
    def compute_energy(sheet, full_output=False):
        nrj_norm_factor = sheet.specs['settings']['nrj_norm_factor']

        E_r = sheet.vert_df.eval('height * radial_tension * is_active')
        if full_output:
            energies = AnchoredSheetModel.compute_energy(sheet,
                                                         full_output=True)
            return energies + (E_r / nrj_norm_factor,)
        else:
            E_base = SheetModel.compute_energy(sheet, full_output=False)
            return E_base + E_r.sum() / nrj_norm_factor

    @staticmethod
    def compute_gradient(sheet, components=False):

        nrj_norm_factor = sheet.specs['settings']['nrj_norm_factor']
        base_grad = AnchoredSheetModel.compute_gradient(sheet,
                                                        components=False)

        rad_grad = (_to_3d(sheet.vert_df['radial_tension']) *
                    height_grad(sheet)) / nrj_norm_factor
        if components:
            return base_grad, rad_grad

        return base_grad + rad_grad
