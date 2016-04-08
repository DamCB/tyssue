"""
Specific functions for apoptosis vertex model
"""

from ..utils.utils import _to_3d
import numpy as np
from .sheet_vertex_model import SheetModel
from .sheet_gradients import height_grad
from copy import deepcopy


class SheetApoptosisModel(SheetModel):

    @staticmethod
    def compute_energy(sheet, full_output=False):
        E_r = sheet.vert_df.eval('height * radial_tension * is_active')
        if full_output:
            E_t, E_c, E_v = SheetModel.compute_energy(sheet, full_output=True)
            return E_t, E_c, E_v, E_r / sheet.nrj_norm_factor
        else:
            E_base = SheetModel.compute_energy(sheet, full_output=False)
            return E_base + E_r.sum() / sheet.nrj_norm_factor

    @staticmethod
    def compute_gradient(sheet, components=False):

        base_grad = SheetModel.compute_gradient(sheet,
                                                components=False)

        rad_grad = (_to_3d(sheet.vert_df['radial_tension']) *
                    height_grad(sheet)) / sheet.nrj_norm_factor
        if components:
            return base_grad, rad_grad

        return base_grad + rad_grad
