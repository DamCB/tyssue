from ..utils.utils import to_nd


def length_grad(sheet):
    """returns -(dx/l, dy/l, dz/l), ie grad_i(l_ij))
    """
    grad_lij = -(
        sheet.edge_df[sheet.dcoords] / to_nd(sheet.edge_df["length"], sheet.dim)
    )
    grad_lij.columns = sheet.coords
    return grad_lij
