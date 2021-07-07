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


def save_datasets(store, eptm):
    """Saves the eptithelium data to a zarr store

    Parameters
    ----------
    store: path to a zarr store, or opened store / group
    eptm: Epithelium object

    Returns
    -------
    the store object
    """
    store_ = zarr.open(store, mode="w")
    with store_:
        store_.attrs.update(filter_settings(eptm.settings))
    for key, dset in eptm.datasets.items():
        dset.to_xarray().to_zarr(store, group=key, mode="w")

    return store
