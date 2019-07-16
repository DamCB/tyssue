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
        self.time = 0

        self.datasets = {}
        self.columns = {}
        vcols = sheet.coords + extra_cols["vert"]
        self.vcols = _filter_columns(vcols, sheet.vert_df.columns, "vertex")
        _vert_h = sheet.vert_df[self.vcols].reset_index(drop=False)
        if not "time" in self.vcols:
            _vert_h["time"] = 0
        self.datasets["vert"] = _vert_h
        self.columns["vert"] = self.vcols

        fcols = extra_cols["face"]
        self.fcols = _filter_columns(fcols, sheet.face_df.columns, "face")
        _face_h = sheet.face_df[self.fcols].reset_index(drop=False)
        if not "time" in self.fcols:
            _face_h["time"] = 0
        self.datasets["face"] = _face_h
        self.columns["face"] = self.fcols

        if sheet.cell_df is not None:
            ccols = extra_cols["cell"]
            self.ccols = _filter_columns(ccols, sheet.cell_df.columns, "cell")
            _cell_h = sheet.cell_df[self.ccols].reset_index(drop=False)
            if not "time" in self.ccols:
                _cell_h["time"] = 0
            self.datasets["cell"] = _cell_h
            self.columns["cell"] = self.ccols
            extra_cols["edge"].append("cell")

        ecols = ["srce", "trgt", "face"] + extra_cols["edge"]
        self.ecols = _filter_columns(ecols, sheet.edge_df.columns, "edge")
        _edge_h = sheet.edge_df[self.ecols].reset_index(drop=False)
        if not "time" in self.ecols:
            _edge_h["time"] = 0
        self.datasets["edge"] = _edge_h
        self.columns["edge"] = self.ecols

    def __len__(self):
        return self.time_stamps.__len__()

    @property
    def time_stamps(self):
        return self.datasets["vert"]["time"].unique()

    @property
    def vert_h(self):
        return self.datasets["vert"]

    @property
    def edge_h(self):
        return self.datasets["edge"]

    @property
    def face_h(self):
        return self.datasets["face"]

    @property
    def cell_h(self):
        return self.datasets.get("cell", None)

    def record(self, to_record=None, time_stamp=None):
        """Appends a copy of the sheet datasets to the history instance.

        Parameters
        ----------
        to_report : list of strings, default ['vert']
            the datasets from self.sheet to be saved

        """
        if to_record is None:
            to_record = ["vert"]

        if time_stamp is not None:
            self.time = time_stamp
        else:
            self.time += 1
        for element in to_record:
            hist = self.datasets[element]
            cols = self.columns[element]
            df = self.sheet.datasets[element][cols].reset_index(drop=False)
            if not "time" in cols:
                times = pd.Series(np.ones((df.shape[0],)) * self.time, name="time")
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

        return type(self.sheet)(
            f"{self.sheet.identifier}_{time:04.3f}", sheet_datasets, self.sheet.specs
        )

    def __iter__(self):

        for t in self.time_stamps:
            sheet = self.retrieve(t)
            yield t, sheet


def _retrieve(dset, time):
    times = dset["time"].values
    t = times[times <= time][-1]
    return dset[dset["time"] == t]
