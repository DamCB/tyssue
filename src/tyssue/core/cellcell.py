import pandas as pd

class CellCellMesh():
    """
    Class to manipulate cell centric models
    """

    def __init__(self, identifier, datasets,
                 specs=None, coords=None):
        '''
        Creates an epithelium

        Parameters:
        -----------
        identifier: string
        datasets: dictionary of dataframes
        the datasets dict specifies the names, data columns
        and value types of the modeled tyssue

        '''
        super().__init__(identifier, datasets,
                         specs, coords)
