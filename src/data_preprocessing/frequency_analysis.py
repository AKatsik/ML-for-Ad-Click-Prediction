""" 
Class to perform frequency analysis.
It can be used for the following:
    - EDA
    - Groupping of categorical features to retain most dominant categories
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from ..config.config_loading import ConfigLoader


class FrequencyAnalyser:
    """ 
    Frequency analysis class:
        - Defines categorical features
        - Run frequency analysis
    """

    def __init__(self,
                 config: ConfigLoader,
                 training_data: pd.DataFrame) -> None:
        self.__config: ConfigLoader = config
        self.__training_data: pd.DataFrame = training_data

        self.__count_unique_vals_in_feature()
        self.__categorical_features = self.__define_categorical_features()

    @property
    def categorical_features(self) -> list:
        """ Categorical features """
        return self.__categorical_features

    def __count_unique_vals_in_feature(self) -> None:
        """ Count the unique values of each feature. """
        print("Number of unique values for each feature")
        print(self.__training_data.nunique())
        print("-" * 40)

    def __define_categorical_features(self) -> list:
        """
        Define categorical features (strings and numeric).
        This is required multiple times later in the pipeline.
        """
        df: pd.DataFrame = self.__training_data.copy()
        non_categ_feat: str = self.__config.generic_config.non_categorical_feaure
        categorical_df: pd.DataFrame = df.loc[:, df.columns != non_categ_feat]
        return list(categorical_df.columns)

    def run_frequency_analysis(self, data: pd.DataFrame) -> dict:
        """ 
        Run frequency analysis and save results.
        The results are used for plotting and entropy calculation.
        """
        frequency_analysis: dict = {}

        for col in data.columns:
            if col in self.__categorical_features:
                val_count: pd.Series = data[col].value_counts(normalize=True)

                frequency_analysis[col] = val_count

        return frequency_analysis

    def plot_frequency_analysis(self, frequency_analysis: dict) -> None:
        """
        Perform frequency analysis for the categorical features.
        In other words, count the occurrences of each unique value in each categorical feature.
        Create bar chart to plot frequency analysis for each feature.
        """
        
        for key, values in frequency_analysis.items():

            print(f"Value counts for '{key}':")
            print(values)
            print("-" * 40)

            # Limit value count to 30 elements to allow visualisation
            val_count_to_plot: pd.Series = values.iloc[:30]

            val_count_to_plot.plot(
                kind="bar",
                figsize=(8, 5)
            )
            plt.title(f"Frequency plot - {key}")
            plt.xlabel(f"{key} - categories")
            plt.ylabel("count")
            plt.xticks(rotation=90)
            plt.tight_layout()

            plt.savefig(os.path.join("plots", f"{key}{len(val_count_to_plot)}.jpg"), dpi=300)

            if self.__config.generic_config.show_plots:
                plt.show()
