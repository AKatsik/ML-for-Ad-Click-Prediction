""" Data loading  into the ML pipeline from the data directory. """

import os
import pandas as pd
from ..config.config_loading import ConfigLoader
from ..data_preprocessing.data_exploration import DataExplorator


class DataLoader(object):

    """
    Load data from the csv file that is defined in the config file.
    """

    def __init__(self, config: ConfigLoader):
        self.__config = config
        self.__data: pd.DataFrame = pd.read_csv(
            os.path.join(
                config.data_paths_elements.data_folder,
                config.data_paths_elements.data_file
            )
        )
        self.__unseen_data = pd.read_csv(
            os.path.join(
                config.data_paths_elements.data_folder,
                config.data_paths_elements.final_test_file
            )
        )

    @property
    def config(self):
        """ Config attribute. """
        return self.__config

    @property
    def data(self):
        """ Data attribute. """
        return self.__data

    def run_data_exploration(self) -> DataExplorator:
        """ Run Exploratory Data Analysis. """
        return DataExplorator(
            data=self.__data,
            config=self.__config,
            unseen_data=self.__unseen_data
        )
