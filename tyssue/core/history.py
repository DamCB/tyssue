import os
import warnings
import traceback
import pandas as pd
import numpy as np

from pathlib import Path

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

    def __init__(self, sheet, save_every=None, dt=None, extra_cols=None, save_all=True):
        """Creates a `SheetHistory` instance.

        Parameters
        ----------
        sheet : a :class:`Sheet` object which we want to record
        save_every : float, set every time interval to save the sheet
        dt : float, time step
        extra_cols : dictionnary with sheet.datasets as keys and list of
            columns as values. Default None
        save_all : bool
            if True, saves all the data at each time point
        """
        if extra_cols is None:
            if save_all:
                extra_cols = {
                    k: list(sheet.datasets[k].columns) for k in sheet.datasets
                }
            else:
                extra_cols = defaultdict(list)
        else:
            extra_cols = defaultdict(list, **extra_cols)

        self.sheet = sheet

        self.time = 0.0
        self.index = 0
        if save_every is not None:
            self.save_every = save_every
            self.dt = dt
        else:
            self.save_every = None

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

    def to_archive(self, hf5file):
        """Saves the history to a HDF file

        This file can later be accessed again with the `HistoryHdf5.from_archive`
        class method

        """
        with pd.HDFStore(hf5file, "a") as store:

            for key, df in self.datasets.items():
                kwargs = {"data_columns": ["time"]}
                if "segment" in df.columns:
                    kwargs["min_itemsize"] = {"segment": 7}
                store.append(key=key, value=df, **kwargs)

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
        to_report : deprecated
        """
        if to_record is not None:
            warnings.warn("Deprecated all the data will be saved")

        if time_stamp is not None:
            self.time = time_stamp
        else:
            self.time += 1

        if (self.save_every is None) or (
            self.index % (int(self.save_every / self.dt)) == 0
        ):
            for element in self.datasets:
                hist = self.datasets[element]
                cols = self.columns[element]
                df = self.sheet.datasets[element][cols].reset_index(drop=False)
                if not "time" in cols:
                    times = pd.Series(np.ones((df.shape[0],)) * self.time, name="time")
                    df = pd.concat([df, times], ignore_index=False, axis=1, sort=False)
                if self.time in hist["time"]:
                    # erase previously recorded time point
                    hist = hist[hist["time"] != self.time]

                hist = pd.concat([hist, df], ignore_index=True, axis=0, sort=False)

                self.datasets[element] = hist

        self.index += 1

    def retrieve(self, time):
        """Return datasets at time `time`.

        If a specific dataset was not recorded at time time, the closest record before that
        time is used.
        """
        if time > self.datasets["vert"]["time"].values[-1]:
            warnings.warn(
                """
The time argument you requested is bigger than the maximum recorded time,
are you sure you pass time in parameter and not an index ?
"""
            )
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

    def __init__(
        self,
        sheet=None,
        save_every=None,
        dt=None,
        extra_cols=None,
        hf5file="",
        overwrite=False,
    ):
        """Creates a `SheetHistory` instance.

        Parameters
        ----------
        sheet : a :class:`Sheet` object which we want to record
        save_every : float, set every time interval to save the sheet
        dt : float, time step
        extra_cols : dictionnary with sheet.datasets as keys and list of
            columns as values. Default None
        hf5file : string, define the path of the HDF5 file
        overwrite : bool, Overwrite or not the file if it is already exist. Default False
        """
        if not hf5file:
            warnings.warn(
                "No directory is given. The HDF5 file will be saved in the working directory as out.hf5."
            )
            self.hf5file = Path(os.getcwd()) / "out.hf5"
        else:
            self.hf5file = Path(hf5file)

        if self.hf5file.exists():
            if overwrite:
                tb = traceback.extract_stack(limit=2)
                if not "from_archive" in tb[0].name:
                    warnings.warn(
                        "The file already exist and will be overwritten."
                        " This is normal if you reopened an archive"
                    )
            else:
                expand = 0
                while True:
                    new_hf5file = self.hf5file.parent / self.hf5file.name.replace(
                        self.hf5file.suffix, f"{expand}{self.hf5file.suffix}"
                    )
                    expand += 1
                    if new_hf5file.exists():
                        continue
                    else:
                        self.hf5file = new_hf5file
                        warnings.warn(
                            "The file already exist and the new filename is {}".format(
                                new_hf5file
                            )
                        )
                        break
        if sheet is None:
            last = self.time_stamps[-1]
            with pd.HDFStore(self.hf5file, "r") as file:
                keys = file.keys()
            if "\cell" in keys:
                sheet = Epithelium

        History.__init__(self, sheet, save_every, dt, extra_cols)
        self.dtypes = {
            k: df[self.columns[k]].dtypes for k, df in sheet.datasets.items()
        }

    @classmethod
    def from_archive(cls, hf5file, columns=None, eptm_class=None):
        datasets = {}
        settings = {}
        hf5file = Path(hf5file)
        with pd.HDFStore(hf5file, "r") as store:
            keys = [k.strip("/") for k in store.keys()]
            if columns is None:
                # read everything
                columns = {k: None for k in keys}

            if eptm_class is None:
                eptm_class = Epithelium if "cell" in keys else Sheet

            last = store.select("vert", columns=["time"]).iloc[-1]["time"]
            for key in keys:
                if key == "settings":
                    settings = store[key]
                    continue
                df = store.select(key, where=f"time == {last}", columns=columns[key])
                datasets[key] = df

        eptm = eptm_class(hf5file.name, datasets)
        eptm.settings.update(settings)
        return cls(sheet=eptm, hf5file=hf5file, overwrite=True)

    @property
    def time_stamps(self):
        with pd.HDFStore(self.hf5file, "r") as file:
            times = file.select("vert", columns=["time"])["time"].unique()
        return times

    def record(self, to_record=None, time_stamp=None, sheet=None):
        """Appends a copy of the sheet datasets to the history HDF file.

        Parameters
        ----------
        to_report : Deprecated - list of strings
            the datasets from self.sheet to be saved
        sheet: a :class:`Sheet` object which we want to record. This argument can
        be used if the sheet object is different at each time point.

        """
        if to_record is not None:
            warnings.warn("Deprecated, all the datasets will be saved anyway")

        if sheet is not None:
            self.sheet = sheet

        if time_stamp is not None:
            self.time = time_stamp
        else:
            self.time += 1.0

        dtypes_ = {k: df.dtypes for k, df in self.sheet.datasets.items()}

        for element, df in self.sheet.datasets.items():
            diff_col = set(dtypes_[element].keys()).difference(
                set(self.dtypes[element].keys())
            )
            if diff_col:
                warnings.warn(
                    "New columns {} will not be saved in the {} table".format(
                        diff_col, element
                    )
                )
            else:
                old_types = self.dtypes[element].to_dict()
                new_types = dtypes_[element].to_dict()
                if new_types != old_types:
                    changed_type = {
                        k: old_types[k]
                        for k in old_types
                        if k in new_types and old_types[k] != new_types[k]
                    }
                    raise ValueError(
                        "There is a change of datatype in {} table in {} columns".format(
                            element, changed_type
                        )
                    )

        if (self.save_every is None) or (
            self.index % (int(self.save_every / self.dt)) == 0
        ):
            for element, df in self.sheet.datasets.items():
                times = pd.Series(np.ones((df.shape[0],)) * self.time, name="time")
                df = df[self.columns[element]]
                df = pd.concat([df, times], ignore_index=False, axis=1, sort=False)
                kwargs = {"data_columns": ["time"]}
                if "segment" in df.columns:
                    kwargs["min_itemsize"] = {"segment": 8}
                with pd.HDFStore(self.hf5file, "a") as file:
                    file.append(key=element, value=df, **kwargs)

        self.index += 1

    def retrieve(self, time):
        """Return datasets at time `time`.

        If a specific dataset was not recorded at time time, the closest record before that
        time is used.
        """
        times = self.time_stamps
        if time > times[-1]:
            warnings.warn(
                "The time argument you passed is bigger than the maximum recorded time, are you sure you pass time in parameter and not an index ? "
            )

        time = times[np.argmin(np.abs(times - time))]
        with pd.HDFStore(self.hf5file, "r") as store:
            sheet_datasets = {}
            for element in self.datasets:
                sheet_datasets[element] = store.select(element, where=f"time == {time}")

        return type(self.sheet)(
            f"{self.sheet.identifier}_{time:04.3f}", sheet_datasets, self.sheet.specs
        )


def _retrieve(dset, time):
    times = dset["time"].values
    t = times[np.argmin(np.abs(times - time))]
    return dset[dset["time"] == t]
