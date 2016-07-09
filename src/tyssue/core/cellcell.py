import pandas as pd
from .objects import Epithelium


class CellCellMesh(Epithelium):
    """
    Class to manipulate cell centric models
    """

    def __init__(self, identifier, datasets,
                 specs=None, coords=None):
        '''

        Parameters:
        -----------
        identifier: string
        datasets: dictionary of dataframes
        the datasets dict specifies the names, data columns
        and value types of the modeled tyssue

        '''
        super().__init__(identifier, datasets,
                         specs, coords)
