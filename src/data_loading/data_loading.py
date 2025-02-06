""" Loading data into the ML pipeline based on the given path. """

import pandas as pd


class DataLoader:

    """ Load data based on the given path. """

    def __init__(self, path: str):

        # Data for training
        self.__loaded_data: pd.DataFrame = pd.read_csv(
            path
        )

    @property
    def loaded_data(self) -> pd.DataFrame:
        """ Loading data. """
        return self.__loaded_data.copy()
