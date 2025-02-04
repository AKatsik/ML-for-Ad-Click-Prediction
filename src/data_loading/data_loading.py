""" Loading data into the ML pipeline from the data directory. """

import os
import copy
import pandas as pd
from ..config.config_loading import ConfigLoader


class DataLoader:

    """
    Load training data and inference data from the csv files,
    whose location is defined in the config file.
    """

    def __init__(self, config: ConfigLoader):
        # Configurations
        self.__config: ConfigLoader = config
        # Data for training
        self.__training_data: pd.DataFrame = pd.read_csv(
            os.path.join(
                config.data_paths_elements.data_folder,
                config.data_paths_elements.data_file
            )
        )
        # Data for inference
        self.__inference_data: pd.DataFrame = pd.read_csv(
            os.path.join(
                config.data_paths_elements.data_folder,
                config.data_paths_elements.final_test_file
            )
        )

    @property
    def config(self) -> ConfigLoader:
        """ Configurations """
        return copy.deepcopy(self.__config)

    @property
    def training_data(self) -> pd.DataFrame:
        """ Data for training """
        return self.__training_data.copy()

    @property
    def inference_data(self) -> pd.DataFrame:
        """ Data for inference """
        return self.__inference_data.copy()
