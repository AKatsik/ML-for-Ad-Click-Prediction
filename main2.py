"""
This is the main of the entire ML pipeline.
AdClassificationExecutor runs the entire pipeline using the given configurations.
"""

import pandas as pd
from src.config.config_loading import ConfigLoader
from src.data_loading.data_loading import DataLoader
from src.data_preprocessing.data_exploration import DataExplorator
from src.model_development.model_training import ModelTrainer
from utils.version import __version__


print(f"Running script1.py - Version: {__version__}")


class ModelExecutor:
    """
    Class to run the entire ML pipeline.
    The only required input is the pipeline configurations.
    """

    def __init__(self, config_path):

        # Load configuration
        self.config: ConfigLoader = ConfigLoader(
            config_path=config_path
        )

        # Load data
        self.data: DataLoader = DataLoader(
            config=self.config
        )

        # Data for training
        self.training_data: pd.DataFrame = self.data.training_data

        # Data for inference
        self.inference_data: pd.DataFrame = self.data.inference_data

        # Exploratory data analysis
        self.eda: DataExplorator = DataExplorator(
            config=self.config,
            training_data=self.training_data,
            inference_data=self.inference_data
        )
        # Processed training data from EDA
        self.processed_train_data: pd.DataFrame = self.eda.training_data

        # Processed inference data from EDA
        self.processed_inference_data: pd.DataFrame = self.eda.inference_data

        # # Model training
        # training: ModelTrainer = ModelTrainer(
        #     config=self.config,
        #     processed_train_data=self.processed_train_data
        # )


if __name__ == "__main__":
    CONFIG_PATH = "config.yaml"
    run = ModelExecutor(config_path=CONFIG_PATH)
