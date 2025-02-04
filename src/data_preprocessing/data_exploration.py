""" Exploratory Data Analysis & Data Preprocessing """

import os
import copy
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
from ..config.config_loading import ConfigLoader
from ..data_preprocessing.frequency_analysis import FrequencyAnalyser


class DataExplorator:
    """
    Perform the following:
    - Exploratory Data Analysis
    - Data cleaning
    - Data preprocessing
    """

    def __init__(self,
                 config: ConfigLoader,
                 training_data: pd.DataFrame,
                 inference_data: pd.DataFrame) -> None:
        
        # Configurations
        self.__config: ConfigLoader = config
        self.__training_data: pd.DataFrame = training_data
        self.__inference_data: pd.DataFrame = inference_data

        # Prepare pipeline required directories
        os.makedirs("plots", exist_ok=True)

        # Data understanding
        self.__show_data_sample()
        self.__show_data_info()
        self.__show_revenue_description()

        # Frequency analysis
        freq_anal = FrequencyAnalyser(
            config=self.__config,
            training_data=self.__training_data
        )
        self.__categorical_features: list = freq_anal.categorical_features
        self.__freq_anal_results: dict = freq_anal.run_frequency_analysis(
            data=self.__training_data
        )
        freq_anal.plot_frequency_analysis(
            frequency_analysis=self.__freq_anal_results
        )

        # Missing values
        self.__analyse_missing_values()
        self.__filter_missing_with_event()
        self.__drop_missing_values()

        # Duplicated rows
        self.__analyse_duplicated_rows()
        self.__remove_duplicated_rows()

        # Strings are encoded to numeric
        self.__calc_categorical_feature_correlation()

        # Remove unused data features from training and inference data
        self.__remove_unused_features()

        # # Group categorical features
        self.__group_rare_categories()
        self.__freq_anal_grouped = freq_anal.run_frequency_analysis(
            data=self.__training_data
        )
        freq_anal.plot_frequency_analysis(
            frequency_analysis=self.__freq_anal_grouped
        )

    @property
    def config(self) -> ConfigLoader:
        """ Config attribute. """
        return copy.deepcopy(self.__config)

    @property
    def training_data(self) -> pd.DataFrame:
        """ Data attribute. """
        return self.__training_data.copy()

    @property
    def inference_data(self) -> pd.DataFrame:
        """ Unseen data attribute. """
        return self.__inference_data.copy()

    def __show_data_sample(self) -> None:
        """ Show head of training and inference data. """
        # Training data
        print("Training Data sample - first 10 rows")
        print(self.__training_data.head(10))
        print("-" * 40)
        # Inference data
        print("Inference Data sample - first 10 rows")
        print(self.__inference_data.head(10))
        print("-" * 40)

    def __show_data_info(self) -> None:
        """ 
        Show data info (types and non-null values) 
        of the training and inference data. 
        """
        # Training data
        print("Training Data Info")
        print(self.__training_data.info())
        print("-" * 40)
        # Inference data
        print("Inference Data Info")
        print(self.__inference_data.info())
        print("-" * 40)

    def __show_revenue_description(self) -> None:
        """
        Show descriptive statistics for revenue feature only,
        which is the only feature with float type.
        """
        print("Revenue - Descriptive statistics")
        print(self.__training_data.revenue.describe())
        print("-" * 40)

    def __analyse_missing_values(self) -> None:
        """
        Analyse missing values in the training data:
            1. Count missing values for each feature.
            2. Calculate the pecentage of missing values for each feature.
            3. Make a dataframe of the results (count & percentage)
            4. Show the dataframe (results)
        """
        # Count and calculate percentage
        missing_count: pd.Series = self.training_data.isnull().sum()
        missing_percentage: pd.Series = (missing_count / len(self.__training_data)) * 100

        # Combine into a DataFrame
        missing_df = pd.DataFrame({
            'Missing Count': missing_count,
            'Missing Percentage': missing_percentage
        })

        # Format the percentage for better readability
        missing_df['Missing Percentage'] = missing_df['Missing Percentage'].round(2)

        print("Missing values analysis")
        print(missing_df)
        print("-" * 40)

    def __filter_missing_with_event(self) -> None:
        """ 
        Count the number of missing values that correspond to the dominant class 0.
        The analysis of the categorical features reveals that the data is highly imbalanced.
        """
        df: pd.DataFrame = self.__training_data.copy()
        missing_and_zero: pd.DataFrame = df[(df["domain"].isnull()) & (df["event_type"] == 1)]

        print("Missing values with zero (0) event")
        print(f"{len(missing_and_zero)} out of {len(df[df["domain"].isnull()])}")
        print("missing values correspond to the dominant class 0")
        print("-" * 40)

    def __drop_missing_values(self) -> None:
        """ 
        From the training data - Drop all missing values using the following logic.
            Only the domain feature contains missing values
            and the vast majority of them correspond to the dominant class (event_type 0).
            In addition, the domain missing values represent the almost the 13% of the feature.
        
        From the inference data - Drop all missing values - No further analysis is performed.
        """
        # Data used for training and test
        self.__training_data.dropna(inplace=True)

        # Unseedn data for final test
        self.__inference_data.dropna(inplace=True)

    def __analyse_duplicated_rows(self) -> None:
        """ 
        Count duplicated rows.
        Calculate the representation of duplicated rows on the dominant class
        """
        duplicated_rows: pd.DataFrame = self.__training_data[self.__training_data.duplicated()]
        dupl_and_zero: pd.DataFrame = duplicated_rows[duplicated_rows["event_type"] == 0]

        print("Duplicates rows & Duplicated rows with zero (0) event")
        print(f"Duplicated rows count: {len(duplicated_rows)}")
        print(f"{len(dupl_and_zero)} out of {len(duplicated_rows)}")
        print("duplicated rows correspond to the dominant class 0")
        print("-" * 40)

    def __remove_duplicated_rows(self) -> None:
        """ 
        Remove all duplicated rows from the training data,
        as all of them represent the dominant class.
        """
        self.__training_data.drop_duplicates(inplace=True)

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

    def __calc_categorical_feature_correlation(self):
        """
        Extract only catgorical features from the data.
        Prepare matrix.
        Create 
        """
        # Prepare an empty matrix
        matrix: pd.DataFrame = pd.DataFrame(
            index=self.__categorical_features,
            columns=self.__categorical_features
        )

        # Calculate correlation for each pair of features in the training data
        for col1 in self.__categorical_features:
            for col2 in self.__categorical_features:

                # Fill in the matrix with Cramer's V
                matrix.loc[col1, col2] = self.__calculate_cramers_v(
                    self.__training_data[col1],
                    self.__training_data[col2]
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

    def __remove_unused_features(self) -> None:
        """ 
        Remove unused features from training and inference data based on the following logic:
            The correlation analysis revealed highly correlated features.
            Only one of these features is retained.
            For now the feature with the more categories is removed.
            
            We keep the features with less categories because we assume that it provides a
            better overview of the data and the current task works like a PoC.
            
            In a later phase, when we would like to improve model performance, we could try
            to tarin models using the features with more categories (more details provided)
            or we could also try to combine the two features to create a new one.
        """

        # Remove from training data
        self.__training_data.drop(
            columns=self.__config.ml_config.features_removed,
            inplace=True
        )

        # Remove from inference data
        self.__inference_data.drop(
            columns=self.__config.ml_config.features_removed,
            inplace=True
        )

    def __group_rare_categories(self) -> None:
        """
        Groups less frequent categories in categorical columns into 'Other'
        if they contribute to less than the specified threshold.
        """
        threshold = self.__config.generic_config.grouping_threshold

        for col in self.__training_data:
            if col in self.__categorical_features:
                # Cumulative sum of frequency
                cumsum = self.__freq_anal_results[col].cumsum()

                # Ensure at least one category is retained
                retained_categories = self.__freq_anal_results[col][
                    (cumsum <= threshold) | (cumsum == cumsum.min())
                ].index

                self.__training_data[col] = self.__training_data[col].apply(
                    lambda x: x if x in retained_categories else "Other"
                )
