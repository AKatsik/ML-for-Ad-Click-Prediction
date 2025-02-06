""" Exploratory Data Analysis & Data Preprocessing """

import os
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from ..config.config_loading import ConfigLoader

def run_frequency_analysis(
    data: pd.DataFrame,
    categorical_features: list
) -> dict:
    """ Run frequency analysis for each categorical feature of the given data. """
    frequency_analysis: dict = {}

    for col in data.columns:
        if col in categorical_features:
            val_count: pd.Series = data[col].value_counts(normalize=True)

            frequency_analysis[col] = val_count

    return frequency_analysis


class DataExplorator:
    """
    Perform the following:
    - Exploratory Data Analysis
    - Data cleaning
    - Data preprocessing
    """

    def __init__(self,
                 config: ConfigLoader,
                 data: pd.DataFrame) -> None:

        # Configurations
        self.__config: ConfigLoader = config
        self.__data: pd.DataFrame = data

        # Prepare pipeline required directories
        os.makedirs("plots", exist_ok=True)

        # --- Data understanding
        self.__show_data_sample()
        self.__show_data_info()
        self.__show_revenue_description()

        # --- Missing values
        self.__analyse_missing_values()

        # --- Frequency analysis
        self.__categorical_features: list = []
        self.__count_unique_vals_in_feature()
        self.__define_categorical_features()

    @property
    def config(self) -> ConfigLoader:
        """ Config property. """
        return copy.deepcopy(self.__config)

    @property
    def processed_data(self) -> pd.DataFrame:
        """ Data property. """
        return self.__data.copy()

    @property
    def categorical_features(self) -> list:
        """ categorical features property. """
        return copy.deepcopy(self.__categorical_features)

    def __show_data_sample(self) -> None:
        """ Show head of given data. """

        print("-" * 10 + " Data sample " + "-" * 10)
        print(self.__data.head(10))
        print("-" * 80)

    def __show_data_info(self) -> None:
        """ Show data info of given data. """

        print("-" * 10 + " Data Info " + "-" * 10)
        print(self.__data.info())
        print("-" * 80)

    def __show_revenue_description(self) -> None:
        """
        Show descriptive statistics for revenue feature only,
        which is the only feature with float type.
        """
        if "revenue" in self.__data.columns:
            print("-" * 10 + " Revenue - Descriptive statistics " + "-" * 10)
            print(self.__data.revenue.describe())
            print("-" * 80)

    def __analyse_missing_values(self) -> None:
        """
        Analyse missing values in the given data:
            1. Count missing values for each feature.
            2. Calculate the pecentage of missing values for each feature.
            3. Make a dataframe of the results (count & percentage)
            4. Show the dataframe (results)
        """
        # Count and calculate percentage
        missing_count: pd.Series = self.__data.isnull().sum()
        missing_percentage: pd.Series = (missing_count / len(self.__data)) * 100

        # Combine into a DataFrame
        missing_df = pd.DataFrame({
            'Missing Count': missing_count,
            'Missing Percentage': missing_percentage
        })

        # Format the percentage for better readability
        missing_df['Missing Percentage'] = missing_df['Missing Percentage'].round(2)

        print("-" * 10 + " Missing values analysis " + "-" * 10)
        print(missing_df)
        print("-" * 80)

    def filter_missing_with_event(self) -> None:
        """
        Count the number of missing values that correspond to the dominant class 0.
        The analysis of the categorical features reveals that the data is highly imbalanced.
        """
        df: pd.DataFrame = self.__data.copy()
        missing_and_zero: pd.DataFrame = df[(df["domain"].isnull()) & (df["event_type"] == 1)]

        print("-" * 10 + " Missing values with zero (0) event " + "-" * 10)
        print(f"{len(missing_and_zero)} out of {len(df[df["domain"].isnull()])}")
        print("missing values correspond to the dominant class 0")
        print("-" * 80)

    def drop_missing_values(self) -> None:
        """ Drop missing values using the following logic. """
        self.__data.dropna(inplace=True)

    def analyse_duplicated_rows(self) -> None:
        """
        Count duplicated rows.
        Calculate the representation of duplicated rows on the dominant class
        """
        duplicated_rows: pd.DataFrame = self.__data[self.__data.duplicated()]
        dupl_and_zero: pd.DataFrame = duplicated_rows[duplicated_rows["event_type"] == 0]

        print("-" * 10 + "Duplicates rows & Duplicated rows with zero (0) event" + "-" * 10)
        print(f"Duplicated rows count: {len(duplicated_rows)}")
        print(f"{len(dupl_and_zero)} out of {len(duplicated_rows)}")
        print("duplicated rows correspond to the dominant class 0")
        print("-" * 80)

    def remove_duplicated_rows(self) -> None:
        """ Remove all duplicated rows. """
        self.__data.drop_duplicates(inplace=True)

    def __count_unique_vals_in_feature(self) -> None:
        """ Count the unique values of each feature. """
        print("-" * 10 + "Number of unique values for each feature" + "-" * 10)
        print(self.__data.nunique())
        print("-" * 80)

    def __define_categorical_features(self) -> None:
        """ Define categorical features. """
        self.__categorical_features = list(self.__data.columns)
        self.__categorical_features.remove(
            self.__config.generic_config.non_categorical_feaure
        )

    def plot_frequency_analysis(self, frequency_analysis: dict) -> None:
        """ Create bar chart to plot frequency analysis for each feature. """

        for key, values in frequency_analysis.items():

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

            # Close plt session
            plt.close()

    def __calculate_cramers_v(self, var1: pd.Series, var2: pd.Series):
        """ Calculate Cramer's V. """
        # Create a contingency table
        contingency_table:pd.DataFrame = pd.crosstab(var1, var2)

        # Perform chi-squared test on the contingency table
        chi2: float
        chi2, _, _, _ = chi2_contingency(contingency_table)

        # Calculate Cramer's V
        n: int = contingency_table.sum().sum()
        cramers_v: np.ndarray = np.sqrt(
            chi2 / (n * (min(contingency_table.shape) - 1))
        )

        return float(cramers_v)

    def calc_categorical_feature_correlation(self):
        """
        Extract only catgorical features from the data.
        Prepare matrix.
        """
        # Remove non-categorical features
        cat_features = self.__categorical_features

        # Prepare an empty matrix
        matrix: pd.DataFrame = pd.DataFrame(
            index=cat_features,
            columns=cat_features
        )

        # Calculate correlation for each pair of features in the training data
        for col1 in cat_features:
            for col2 in cat_features:

                # Fill in the matrix with Cramer's V
                matrix.loc[col1, col2] = self.__calculate_cramers_v(
                    self.__data[col1],
                    self.__data[col2]
                )

        # Convert matrix data into numeric
        matrix: pd.DataFrame = matrix.apply(pd.to_numeric, errors='coerce')

        # Create the heatmap
        plt.figure(figsize=(8, 8))
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            cbar=True,
            annot_kws={"size": 8}
        )

        # Add titles and labels
        # plt.title("Categorical feature correlation with Cramer's V")
        plt.tight_layout()

        # Remove the comment below to save the plot
        plt.savefig(os.path.join("plots",'correlation.png'), transparent=True)

        if self.__config.generic_config.show_plots:
            plt.show()

        # Close plt session
        plt.close()
