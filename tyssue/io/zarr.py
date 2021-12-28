import os
import pandas as pd
import logging
import warnings

try:
    import xarray as xr
    import zarr
except ImportError:
    warnings.warn(
        """You need the zarr and xarray packages to use this module,
the load and save functions will not work.
"""
    )

from .utils import filter_settings

logger = logging.getLogger(name=__name__)


def load_datasets(store):
    """Loads an epithelium dataset and settings from a zarr store

    Parameters
    ----------
    store: path to a zarr store, or opened store / group

    Returns
    -------
    datasets: dictionary of pd.DataFrame objects
    settings: dictionnary

    """
    with zarr.open(store, mode="r") as store_:
        settings = dict(store_.attrs)
        keys = store_.group_keys()

    datasets = {key: xr.open_zarr(store, key).to_dataframe() for key in keys}

    return datasets, settings


def save_datasets(store, eptm, grp=None):
    """Saves the eptithelium data to a zarr store

    Parameters
    ----------
    store: path to a zarr store, or opened store / group
    eptm: an Epithelium object
    grp: optional, str
        name of a group within the store

    Returns
    -------
    the store object
    """
    if grp:
        root = zarr.group(store)
        group = root.create_group(grp, overwrite=True)

    with zarr.open(store, mode="w") as store_:
        store_.attrs.update(filter_settings(eptm.settings))

    for key, dset in eptm.datasets.items():
        if grp:
            group.create_group(key)
            key = f"{grp}/{key}"
        dset.to_xarray().to_zarr(store, group=key, mode="w")

    return store
