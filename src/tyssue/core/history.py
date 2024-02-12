import logging
import os
import traceback
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .objects import Epithelium
from .sheet import Sheet

logger = logging.getLogger(name=__name__)


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
    """This class handles recording and retrieving time series
    of sheet objects.

    """

    def __init__(
        self,
        sheet,
        save_every=None,
        dt=None,
        save_only=None,
        extra_cols=None,
        save_all=True,
    ):
        """Creates a `SheetHistory` instance.

        Parameters
        ----------
        sheet : a :class:`Sheet` object which we want to record
        save_every : float, set every time interval to save the sheet
        dt : float, time step
        save_only: dict : dictionnary with sheet.datasets as keys and list of
            columns as values. Default None

        extra_cols : dictionnary with sheet.datasets as keys and list of
            columns as values. Default None
        save_all : bool
            if True, saves all the data at each time point

        """
        if extra_cols is not None:
            warnings.warn(
                "extra_cols and save_all parameters are deprecated."
                " Use save_only instead. "
            )

        if save_only is not None:
            extra_cols = defaultdict(list, **save_only)
        else:
            extra_cols = {k: list(sheet.datasets[k].columns) for k in sheet.datasets}

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
        if "time" not in self.vcols:
            _vert_h["time"] = 0
        self.datasets["vert"] = _vert_h
        self.columns["vert"] = self.vcols

        fcols = extra_cols["face"]
        self.fcols = _filter_columns(fcols, sheet.face_df.columns, "face")
        _face_h = sheet.face_df[self.fcols].reset_index(drop=False)
        if "time" not in self.fcols:
            _face_h["time"] = 0
        self.datasets["face"] = _face_h
        self.columns["face"] = self.fcols

        if sheet.cell_df is not None:
            ccols = extra_cols["cell"]
            self.ccols = _filter_columns(ccols, sheet.cell_df.columns, "cell")
            _cell_h = sheet.cell_df[self.ccols].reset_index(drop=False)
            if "time" not in self.ccols:
                _cell_h["time"] = 0
            self.datasets["cell"] = _cell_h
            self.columns["cell"] = self.ccols
            extra_cols["edge"].append("cell")

        ecols = ["srce", "trgt", "face"] + extra_cols["edge"]
        ecols = list(set(ecols))
        self.ecols = _filter_columns(ecols, sheet.edge_df.columns, "edge")
        _edge_h = sheet.edge_df[self.ecols].reset_index(drop=False)
        if "time" not in self.ecols:
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

    def record(self, time_stamp=None):
        """Appends a copy of the sheet datasets to the history instance.

        Parameters
        ----------
        time_stamp : float, save specific timestamp
        """

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
                if "time" not in cols:
                    times = pd.Series(np.ones((df.shape[0],)) * self.time, name="time")
                    df = pd.concat([df, times], ignore_index=False, axis=1, sort=False)
                else:
                    df["time"] = self.time

                if self.time in hist["time"]:
                    # erase previously recorded time point
                    hist = hist[hist["time"] != self.time]

                hist = pd.concat([hist, df], ignore_index=True, axis=0, sort=False)

                self.datasets[element] = hist

        self.index += 1

    def retrieve(self, time):
        """Return datasets at time `time`.

        If a specific dataset was not recorded at time time,
        the closest record before that time is used.
        """
        if time > self.datasets["vert"]["time"].values[-1]:
            warnings.warn(
                """
The time argument you requested is bigger than the maximum recorded time,
are you sure you passed the time stamp as parameter, and not an index ?
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
        """Iterates over all the time points of the history"""
        for t in self.time_stamps:
            sheet = self.retrieve(t)
            yield t, sheet

    def slice(self, start=0, stop=None, size=None, endpoint=True):
        """Returns a slice of the history's time_stamps array

        The slice is over or under sampled to have exactly size point
        between start and stop
        """
        if size is not None:
            if stop is not None:
                time_stamps = self.time_stamps[start : stop + int(endpoint)]
            else:
                time_stamps = self.time_stamps
            indices = np.round(
                np.linspace(0, time_stamps.size + 1, size, endpoint=True)
            ).astype(int)
            times = time_stamps.take(indices.clip(max=time_stamps.size - 1))
        elif stop is not None:
            times = self.time_stamps[start : stop + int(endpoint)]
        else:
            times = self.time_stamps
        return times

    def browse(self, start=0, stop=None, size=None, endpoint=True):
        """Returns an iterator over part of the history

        Parameters
        ----------

        start: int, index of the first time point
        stop: int, index of the last time point
        size: int, the number of time points to return
        endpoint: bool, wether to include the stop time point (default True)

        Returns
        -------
        iterator over (t, sheet(t)) for the retrieved time points


        """
        for t in self.slice(start=start, stop=stop, size=size, endpoint=endpoint):
            yield t, self.retrieve(t)


class HistoryHdf5(History):
    """This class handles recording and retrieving time series
    of sheet objects.

    """

    def __init__(
        self,
        sheet=None,
        save_every=None,
        dt=None,
        save_only=None,
        hf5file="",
        overwrite=False,
    ):
        """Creates a `HistoryHdf5` instance.

        Use the `from_archive` class method to re-open a saved history file

        Parameters
        ----------
        sheet : a :class:`Sheet` object which we want to record
        save_every : float, set every time interval to save the sheet
        dt : float, time step
        save_only : dictionnary with sheet.datasets as keys and list of
            columns as values. Default None
        hf5file : string, define the path of the HDF5 file
        overwrite : bool, Overwrite or not the file if it is already exist.
            Default False
        """
        if not hf5file:
            warnings.warn(
                "No directory is given. The HDF5 file will be saved"
                " in the working directory as out.hf5."
            )
            self.hf5file = Path(os.getcwd()) / "out.hf5"
        else:
            self.hf5file = Path(hf5file)

        if self.hf5file.exists():
            if overwrite:
                tb = traceback.extract_stack(limit=2)
                if "from_archive" not in tb[0].name:
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

        with pd.HDFStore(self.hf5file, "r") as file:
            self._time_stamps = file.select("vert", columns=["time"])["time"].unique()

        if sheet is None:
            last = self.time_stamps[-1]
            with pd.HDFStore(self.hf5file, "r") as file:
                keys = file.keys()
            if r"\cell" in keys:
                sheet = Epithelium("test", last)

        History.__init__(self, sheet, save_every, dt, save_only)
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
    def time_stamps(self, element="vert"):
        return self._time_stamps

    def record(self, time_stamp=None, sheet=None):
        """Appends a copy of the sheet datasets to the history HDF file.

        Parameters
        ----------
        sheet: a :class:`Sheet` object which we want to record. This argument can
        be used if the sheet object is different at each time point.

        """

        if sheet is not None:
            self.sheet = sheet

        if time_stamp is not None:
            self.time = time_stamp
        else:
            self.time += 1.0

        dtypes_ = {k: df.dtypes for k, df in self.sheet.datasets.items()}

        for element, df in self.sheet.datasets.items():
            old_types = self.dtypes[element].to_dict()
            new_types = {k: dtypes_[element].to_dict()[k] for k in old_types.keys()}

            if new_types != old_types:
                changed_type = {
                    k: old_types[k]
                    for k in old_types
                    if k in new_types and old_types[k] != new_types[k]
                }
                raise ValueError(
                    f"There is a change of datatype in {element} table"
                    f" in {changed_type} columns"
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
                with pd.HDFStore(self.hf5file, "a") as store:
                    if (
                        element in store
                        and store.select(element, where=f"time == {self.time}")[
                            "time"
                        ].shape[0]
                        > 0
                    ):
                        store.remove(key=element, where=f"time == {self.time}")
                    store.append(key=element, value=df, **kwargs)

        self.index += 1

    def retrieve(self, time):
        """Returns datasets at time `time`.

        If a specific dataset was not recorded at time time,
        the closest record before that time is used.
        """
        times = self.time_stamps
        if time > times[-1]:
            warnings.warn(
                "The time argument you passed is bigger than the maximum recorded time,"
                " are you sure you pass time in parameter and not an index ? "
            )

        time = times[np.argmin(np.abs(times - time))]
        with pd.HDFStore(self.hf5file, "r") as store:
            sheet_datasets = {}
            for element in self.datasets:
                sheet_datasets[element] = store.select(element, where=f"time == {time}")

        sheet = type(self.sheet)(
            f"{self.sheet.identifier}_{time:04.3f}", sheet_datasets, self.sheet.specs
        )
        sheet.coords = self.sheet.coords
        sheet.edge_df.index.rename("edge", inplace=True)
        return sheet

    def retrieve_columns(self, element, columns):
        """
        Return a table with the selected columns from the given element

        Parameters
        ----------
        element: str
            either 'vert', 'edge', 'face' or 'cell'
        columns: list of str
            a list of columns to retrieve


        """
        with pd.HDFStore(self.hf5file, "r") as store:
            data = store.select(
                element,
                columns=columns,
            )
        return data


def _retrieve(dset, time):
    times = dset["time"].values
    t = times[np.argmin(np.abs(times - time))]
    return dset[dset["time"] == t]
