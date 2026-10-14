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
    root = zarr.open_group(store, mode="r")
    settings = dict(root.attrs)
    keys = list(root.group_keys())

    datasets = {key: xr.open_zarr(store, group=key).to_dataframe() for key in keys}

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
    # overwrites any existing store
    root = zarr.open_group(store, mode="w")
    root.attrs.update(filter_settings(eptm.settings))

    for key, dset in eptm.datasets.items():
        path = f"{grp}/{key}" if grp else key
        dset.to_xarray().to_zarr(store, group=path, mode="a")

    return store
