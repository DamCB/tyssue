import warnings
import pandas as pd
import numpy as np

from collections import defaultdict
from .sheet import Sheet
from .objects import Epithelium


def _filter_columns(cols_hist, cols_in, element):
    if not set(cols_hist).issubset(cols_in):
        warnings.warn(
            f"""
Columns {set(cols_hist).difference(cols_in)} are in the history
 {element} dataframe but not in the sheet {element} dataframe.

These non existent columns will not be saved."""
        )
        cols_hist = set(cols_hist).intersection(cols_in)
    return list(cols_hist)


class History:
    """ This class handles recording and retrieving time series
    of sheet objects.

    """

    def __init__(self, sheet, extra_cols=None):
        """Creates a `SheetHistory` instance.

        Parameters
        ----------
        sheet : a :class:`Sheet` object which we want to record
        extra_cols : dictionnary with sheet.datasets as keys and list of
            columns as values. Default None
        """
        if extra_cols is None:
            extra_cols = defaultdict(list)
        else:
            extra_cols = defaultdict(list, **extra_cols)

        self.sheet = sheet
        self.t = 0

        vcols = sheet.coords + extra_cols["vert"]
        self.vcols = _filter_columns(vcols, sheet.vert_df.columns, "vertex")
        self.vert_h = sheet.vert_df[self.vcols].reset_index(drop=False)
        if not "t" in self.vcols:
            self.vert_h["t"] = 0

        ecols = ["srce", "trgt", "face"] + extra_cols["edge"]
        self.ecols = _filter_columns(ecols, sheet.edge_df.columns, "edge")
        self.edge_h = sheet.edge_df[self.ecols].reset_index(drop=False)
        if not "t" in self.ecols:
            self.edge_h["t"] = 0

        fcols = extra_cols["face"]
        self.fcols = _filter_columns(fcols, sheet.face_df.columns, "face")
        self.face_h = sheet.face_df[self.fcols].reset_index(drop=False)
        if not "t" in self.fcols:
            self.face_h["t"] = 0
        self.datasets = {"vert": self.vert_h, "edge": self.edge_h, "face": self.face_h}
        self.columns = {"vert": self.vcols, "edge": self.ecols, "face": self.fcols}

        if sheet.cell_df is not None:
            ccols = extra_cols["cell"]
            self.ccols = _filter_columns(ccols, sheet.cell_df.columns, "cell")
            self.cell_h = sheet.cell_df[self.ccols].reset_index(drop=False)
            if not "t" in self.ccols:
                self.cell_h["t"] = 0
            self.datasets["cell"] = self.cell_h
            self.columns["cell"] = self.ccols

    def record(self, to_record=["vert"]):
        """Appends a copy of the sheet datasets to the history instance.

        Parameters
        ----------
        to_report : list of strings, default ['vert']
            the datasets from self.sheet to be saved

        """

        self.t += 1
        for element in to_record:
            hist = self.datasets[element]
            cols = self.columns[element]
            df = self.sheet.datasets[element][cols].reset_index(drop=False)
            if not "t" in cols:
                times = pd.Series(np.ones((df.shape[0],), dtype=int) * self.t, name="t")
                df = pd.concat([df, times], ignore_index=False, axis=1, sort=False)
            hist = pd.concat([hist, df], ignore_index=True, axis=0, sort=False)
            self.datasets[element] = hist

    def retrieve(self, time):
        """Return datasets at time `time`.

        If a specific dataset was not recorded at time time, the closest record before that
        time is used.
        """
        sheet_datasets = {}
        for element in self.datasets:
            hist = self.datasets[element]
            cols = self.columns[element]
            df = _retrieve(hist, time)
            df = df.set_index(element)[cols]
            sheet_datasets[element] = df

        return sheet_datasets


def _retrieve(dset, time):
    times = dset["t"].values
    t = times[times <= time][-1]
    return dset[dset["t"] == t]
