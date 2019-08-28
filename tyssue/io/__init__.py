"""
Input/Output module.

Supported formats:

hdf5, csv
"""
import os

from pathlib import Path


def get_sim_dir():
    """Returns the path to the root simulation directory
    as a :class:`pathlib.Path` object

    The simulation directory is stored in the  'TYSSUE_SIM_DIR'
    environement variable.


    """
    try:
        sim_dir = Path(os.environ["TYSSUE_SIM_DIR"]).absolute()
    except KeyError:
        print(
            """
No default found for the simulations directory,
you can define it by setting the "TYSSUE_SIM_DIR"
environment variable on your system (e.g in .bashrc).

We are now using the current directory as default.
            """
        )
        sim_dir = Path(os.getcwd())

    return sim_dir
