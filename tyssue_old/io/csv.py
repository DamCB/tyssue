import pandas as pd
import numpy as np


def write_storm_csv(
    filename, points, coords=["x", "y", "z"], split_by=None, **csv_args
):
    """
    Saves a point cloud array in the storm format
    """
    columns = ["frame", "x [nm]", "y [nm]", "z [nm]", "uncertainty_xy", "uncertainty_z"]
    points = points.dropna()
    storm_points = pd.DataFrame(np.zeros((points.shape[0], 6)), columns=columns)
    storm_points[["x [nm]", "y [nm]", "z [nm]"]] = points[coords].values
    storm_points["frame"] = 1
    storm_points[["uncertainty_xy", "uncertainty_z"]] = 2.1
    # tab separated values are faster and more portable than excel
    if split_by is None:
        if not filename.endswith(".csv"):
            filename = filename + ".csv"
        storm_points.to_csv(filename, **csv_args)
    elif split_by in points.columns():
        storm_points[split_by] = points[split_by]
        # separated files by the column split_by
        storm_points.groupby(split_by).apply(
            lambda df: df.to_csv(
                "{}_{}.csv".format(filename, df[split_by].iloc[0]), **csv_args
            )
        )
