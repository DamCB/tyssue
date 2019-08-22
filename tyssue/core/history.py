import os
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

    def __init__(self, sheet, save_every=None, dt=None, extra_cols=None):
        """Creates a `SheetHistory` instance.

        Parameters
        ----------
        sheet : a :class:`Sheet` object which we want to record
        save_every : float, set every time interval to save the sheet
        dt : float, time step
        extra_cols : dictionnary with sheet.datasets as keys and list of
            columns as values. Default None
        """
        if extra_cols is None:
            extra_cols = defaultdict(list)
        else:
            extra_cols = defaultdict(list, **extra_cols)

        self.sheet = sheet

        self.time = 0
        self.index = 0
        if save_every is not None:
            self.save_every = save_every
            self.dt = dt

        self.datasets = {}
        self.columns = {}
        vcols = sheet.coords + extra_cols["vert"]
        vcols = list(set(vcols))
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
        ecols = list(set(ecols))
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

        if (self.save_every is None) or (self.index % (int(self.save_every / self.dt)) == 0):
            for element in to_record:
                hist = self.datasets[element]
                cols = self.columns[element]
                df = self.sheet.datasets[element][cols].reset_index(drop=False)
                if not "time" in cols:
                    times = pd.Series(
                        np.ones((df.shape[0],)) * self.time, name="time")
                    df = pd.concat([df, times], ignore_index=False,
                                   axis=1, sort=False)
                if self.time in hist["time"]:
                    # erase previously recorded time point
                    hist = hist[hist["time"] != self.time]

                hist = pd.concat(
                    [hist, df], ignore_index=True, axis=0, sort=False)

                self.datasets[element] = hist

        self.index += 1

    def retrieve(self, time):
        """Return datasets at time `time`.

        If a specific dataset was not recorded at time time, the closest record before that
        time is used.
        """
        if time > self.datasets["vert"]["time"].values[-1]:
            warnings.warn(
                "The time argument you passed is bigger than the maximum recorded time, are you sure you pass time in parameter and not an index ? ")
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


class HistoryHdf5(History):
    """ This class handles recording and retrieving time series
    of sheet objects.

    """

    def __init__(self, sheet, save_every=None, dt=None, extra_cols=None, path="", overwrite=False):
        """Creates a `SheetHistory` instance.

        Parameters
        ----------
        sheet : a :class:`Sheet` object which we want to record
        save_every : float, set every time interval to save the sheet
        dt : float, time step
        extra_cols : dictionnary with sheet.datasets as keys and list of
            columns as values. Default None
        path : string, define the path where save HDF5 file
        overwrite : bool, Overwrite or not the file if it is already exist. Default False
        """
        if path is None:
            warnings.warn(
                "No directory is given. The HDF5 file will be saved in the working directory.")
            self.path = os.os.getcwd()
        else:
            self.path = path

        if os.path.exists(os.path.join(self.path, 'out.hf5')):
            if overwrite:
                self.hf5file = os.path.join(self.path, 'out.hf5')
                pd.HDFStore(self.hf5file, 'w').close()
                warnings.warn(
                    "The file already exist and will be overwritten.")
            else:
                expend = 1
                while True:
                    expend += 1
                    new_file_name = "out{}.hf5".format(str(expend))
                    if os.path.exists(os.path.join(self.path, new_file_name)):
                        continue
                    else:
                        self.hf5file = os.path.join(self.path, new_file_name)
                        pd.HDFStore(self.hf5file, 'w').close()
                        warnings.warn(
                            "The file already exist and the new filename is {}".format(new_file_name))
                        break

        else:
            self.hf5file = os.path.join(self.path, 'out.hf5')
            pd.HDFStore(self.hf5file, 'w').close()
        History.__init__(self, sheet, save_every, dt, extra_cols)

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

        if (self.save_every is None) or (self.index % (int(self.save_every / self.dt)) == 0):
            for element in to_record:
                df = self.sheet.datasets[element]
                times = pd.Series(
                    np.ones((df.shape[0],)) * self.time, name="time")
                df = pd.concat([df, times], ignore_index=False,
                               axis=1, sort=False)

                with pd.HDFStore(self.hf5file, 'a') as file:

                    file.append(key="{}_df".format(element),
                                    value=df)


        self.index += 1

    def retrieve(self, time):
        """Return datasets at time `time`.

        If a specific dataset was not recorded at time time, the closest record before that
        time is used.
        """
        if time > self.datasets["vert"]["time"].values[-1]:
            warnings.warn(
                "The time argument you passed is bigger than the maximum recorded time, are you sure you pass time in parameter and not an index ? ")

        with pd.HDFStore(self.hf5file, 'r') as file:
            sheet_datasets = {}
            for element in self.datasets:
                hist = file["{}_df".format(element)]
                cols = self.columns[element]
                df = _retrieve(hist, time)
                sheet_datasets[element] = df

        return type(self.sheet)(
            f"{self.sheet.identifier}_{time:04.3f}", sheet_datasets, self.sheet.specs
        )


def _retrieve(dset, time):
    times = dset["time"].values
    t = times[np.argmin(np.abs(times - time))]
    return dset[dset["time"] == t]
