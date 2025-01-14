""" Classification model inference.  """

import pandas as pd
from .model_training import ModelTrainer


class ModelPredictor(ModelTrainer):
    """ Use the best model from ModelBase to make predictions for unseen data. """
    def __init__(self, config, data, unseen_data):
        super().__init__(data=data, config=config, unseen_data=unseen_data)  # Initialize ModelBase
        self.__unseen_data = unseen_data

        self.final_y_pred, self.final_y_prob = self.predict(test_data=unseen_data)
        self.save_predictions_to_csv()

    def save_predictions_to_csv(self):
        """Save unseen data, predictions, and binary probabilities to a CSV file."""
        # Create a DataFrame with unseen data and predictions
        data = pd.DataFrame(self.__unseen_data)
        data['final_y_pred'] = self.final_y_pred
        data['binary_prob'] = self.final_y_prob

        # Save the DataFrame to a CSV file
        data.to_csv("final_results.csv", index=False)
