"""
This is the main of the entire ML pipeline.
AdClassificationExecutor runs the entire pipeline using the given configurations.
"""

from src.config.config_loading import ConfigLoader
from src.data_loading.data_loading import DataLoader
from src.model_development.model_inference import ModelPredictor
from utils.version import __version__


print(f"Running script1.py - Version: {__version__}")


class AdClassificationExecutor:
    """
    Class to run the entire ML pipeline.
    The only required input is the pipeline configurations.
    """
    def __init__(self, config_path):
        config = ConfigLoader(config_path)
        self.run = DataLoader(config=config)\
            .run_data_exploration()

        # self.final_result = ModelPredictor(
        #     data=self.run.data,
        #     config=config,
        #     unseen_data=self.run.unseen_data
        # )


if __name__ == "__main__":
    CONFIG_PATH = "config.yaml"
    run = AdClassificationExecutor(config_path=CONFIG_PATH)
