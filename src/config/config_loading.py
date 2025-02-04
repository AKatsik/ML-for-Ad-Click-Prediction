"""
Reading and loading configurations for the ML pipeline.
The original configurations are included in the config.yaml.
"""

from dataclasses import dataclass
import typing as t
import copy
from ..helper.helper import Helper


@dataclass
class DataLocation:
    """ Data paths configurations. """
    data_folder: str
    data_file: str
    final_test_file: str

    @classmethod
    def read_config(cls: t.Type["DataLocation"], obj: dict):
        """ Read the configuration about data paths. """
        return cls(
            data_folder=obj["data_paths"]["data_folder"],
            data_file=obj["data_paths"]["data_file"],
            final_test_file=obj["data_paths"]["final_test_file"]
        )


@dataclass
class GenericConfig:
    """ General configurations for the ML pipeline. """
    non_categorical_feaure: str
    random_state: int
    show_plots: bool
    grouping_threshold: float

    @classmethod
    def read_config(cls: t.Type["GenericConfig"], obj: dict):
        """ Read the general configurations about the ML pipeline. """
        return cls(
            non_categorical_feaure=obj["generic_config"]["non_categorical_feaure"],
            random_state=obj["generic_config"]["random_state"],
            show_plots=obj["generic_config"]["show_plots"],
            grouping_threshold=obj["generic_config"]["grouping_threshold"]
        )


@dataclass
class MlConfig:
    """ Machine learning model configurations. """
    features_removed: list
    test_size: float
    n_estimators: list
    max_depth: list
    min_samples_split: list
    min_samples_leaf: list
    max_features: list
    bootstrap: list
    iterations: int
    cv_folds: int

    @classmethod
    def read_config(cls: t.Type["MlConfig"], obj: dict):
        """ Read the ML model configurations. """
        return cls(
            features_removed=obj["ml_config"]["features_removed"],
            test_size=obj["ml_config"]["test_size"],
            n_estimators=obj["ml_config"]["n_estimators"],
            max_depth=obj["ml_config"]["max_depth"],
            min_samples_split=obj["ml_config"]["min_samples_split"],
            min_samples_leaf=obj["ml_config"]["min_samples_leaf"],
            max_features=obj["ml_config"]["max_features"],
            bootstrap=obj["ml_config"]["bootstrap"],
            iterations=obj["ml_config"]["iterations"],
            cv_folds=obj["ml_config"]["cv_folds"]
        )


class ConfigLoader(object):
    """ Load all the configuations. """

    def __init__(self, config_path):
        config_file = Helper.read_yaml_file(path=config_path)

        self.__data_paths_elements = DataLocation.read_config(obj=config_file)
        self.__generic_config = GenericConfig.read_config(obj=config_file)
        self.__ml_config = MlConfig.read_config(obj=config_file)

    @property
    def data_paths_elements(self):
        """ Data paths """
        return copy.deepcopy(self.__data_paths_elements)

    @property
    def generic_config(self):
        """ Data paths """
        return copy.deepcopy(self.__generic_config)

    @property
    def ml_config(self):
        """ Data paths """
        return copy.deepcopy(self.__ml_config)
