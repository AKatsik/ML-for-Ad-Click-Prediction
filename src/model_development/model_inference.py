""" Classification model inference.  """

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator


class Predictor:
    """ Use the best model from ModelBase to make predictions for unseen data. """
    def __init__(self,
                 classification_model: BaseEstimator,
                 inference_data: pd.DataFrame) -> None:

        self.__inference_data: pd.DataFrame = inference_data

        self.__final_predictions: np.ndarray = classification_model.predict(
            inference_data
        )
        self.__final_probabilities: np.ndarray = classification_model.predict_proba(
            inference_data
        )[:, 1]    

        # self.final_y_pred, self.final_y_prob = self.predict(test_data=unseen_data)
        self.save_predictions_to_csv()

    @property
    def final_prediction(self) -> np.ndarray:
        """ Final prediction property. """
        return self.__final_predictions

    @property
    def final_probabilities(self) -> np.ndarray:
        """ Final probabilities property. """
        return self.__final_probabilities

    def save_predictions_to_csv(self):
        """Save unseen data, predictions, and binary probabilities to a CSV file."""
        # Create a DataFrame with unseen data and predictions
        data = self.__inference_data
        data['final_y_pred'] = self.__final_predictions
        data['binary_prob'] = self.__final_probabilities

        # Save the DataFrame to a CSV file
        data.to_csv("final_results.csv", index=False)
