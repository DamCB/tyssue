from .sheet_geometry import SheetGeometry


class MultiSheetGeometry:
    """ Geometry class for stacked 2D sheets

    """

    @classmethod
    def update_all(cls, msheet):

        msheet.update_interpolants()
        for sheet in msheet:
            SheetGeometry.update_dcoords(sheet)
            SheetGeometry.update_length(sheet)
            SheetGeometry.update_centroid(sheet)
            SheetGeometry.update_normals(sheet)
            SheetGeometry.update_areas(sheet)
            SheetGeometry.update_perimeters(sheet)
        cls.update_heights(msheet)

    @staticmethod
    def update_heights(msheet):

        msheet[0].vert_df["height"] = (
            msheet[0].vert_df["z"] - msheet[0].vert_df["basal_shift"]
        )

        #  Here we use basal_shift to impose apical constrains
        msheet[-1].vert_df["depth"] = (
            msheet[-1].vert_df["basal_shift"] - msheet[-1].vert_df["z"]
        )

        for lower, upper in zip(msheet.interpolants[:-1], msheet[1:]):
            upper.vert_df["height"] = upper.vert_df["z"] - lower(
                upper.vert_df["x"], upper.vert_df["y"]
            )
        for lower, upper in zip(msheet[:-1], msheet.interpolants[1:]):
            lower.vert_df["depth"] = (
                upper(lower.vert_df["x"], lower.vert_df["y"]) - lower.vert_df["z"]
            )
