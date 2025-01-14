""" Exploratory Data Analysis. """

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
from ..config.config_loading import ConfigLoader
from ..model_development.model_training import ModelTrainer


class DataExplorator:
    """
    Perform Exploratory Data Analysis including data cleaning.
    The class executes the following steps in the given order:
        1. Understand the date
        2. Perform frequency analysis for categorical features
        3. Analyse and handle missing values
        4. Analyse and handle duplicated values
        5. Perform correlation analysis
    """

    def __init__(self,
                 data: pd.DataFrame,
                 config: ConfigLoader,
                 unseen_data) -> None:
        self.__config = config
        self.__data = data
        self.__unseen_data = unseen_data

        # Data understanding
        self.__show_data_sample()
        self.__show_data_info()
        self.__show_revenue_description()
        self.__categorical_features = self.__define_categorical_features()
        self.__count_unique_vals_in_feature()
        self.__count_each_unique_val_in_feature()

        # Missing values
        self.__analyse_missing_values()
        self.__filter_missing_with_event()
        self.__drop_missing_values()

        # Duplicated rows
        self.__analyse_duplicated_rows()
        self.__remove_duplicated_rows()

        # # Strings are encoded to numeric
        self.__calc_categorical_feature_correlation()

    @property
    def config(self):
        """ Config attribute. """
        return self.__config

    @property
    def data(self):
        """ Data attribute. """
        return self.__data

    @property
    def unseen_data(self):
        """ Unseen data attribute. """
        return self.__unseen_data

    def __show_data_sample(self) -> None:
        """ Show head of the given data. """
        
        print("Data sample - first 10 rows")
        print(self.__data.head(10))
        print("-" * 40)

    def __show_data_info(self) -> None:
        """ Show data info (types and non-null values). """
        print("Data Info")
        print(self.__data.info())
        print("-" * 40)

    def __show_revenue_description(self) -> None:
        """
        Show descriptive statistics for revenue feature only, which is the only feature with float type.
        """
        print("Revenue - Descriptive statistics")
        print(self.__data.revenue.describe())
        print("-" * 40)

    def __count_unique_vals_in_feature(self) -> None:
        """ Count the unique values of each feature. """
        print("Number of unique values for each feature")
        print(self.__data.nunique())
        print("-" * 40)

    def __count_each_unique_val_in_feature(self) -> None:
        """ 
        Count the occurrences of each unique value in each categorical feature.
        Create bar chart to plot frequency analysis for each feature and save it.
        """
        for col in self.__categorical_features:
            val_count = self.__data[col].value_counts()

            print(f"Value counts for '{col}':")
            print(val_count)
            print("-" * 40)

            # Limit value count to 30 elements to allow visualisation
            val_count_to_plot = val_count.iloc[:30]

            val_count_to_plot.plot(
                kind="bar",
                figsize=(12, 8)
            )
            plt.title(f"Frequency plot - {col}")
            plt.xlabel(f"{col} - categories")
            plt.ylabel("count")
            plt.xticks(rotation=90)
            plt.tight_layout()

            if self.__config.generic_config.show_plots:
                plt.show()

    def __define_categorical_features(self) -> list:
        """
        Define categorical features (strings and numeric).
        This is required multiple times later in the pipeline.
        """
        df = self.__data.copy()
        categorical_df = df.loc[:, df.columns != self.__config.generic_config.non_categorical_feaure]
        return list(categorical_df.columns)

    def __analyse_missing_values(self) -> None:
        """
        Analyse missing values:
            1. Count missing values for each feature.
            2. Calculate the pecentage of missing values for each feature.
            3. Add both factors in a dataframe
            4. Show dataframe
        """
        # Count and calculate percentage
        missing_count = self.__data.isnull().sum()
        missing_percentage = (missing_count / len(self.__data)) * 100
        
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
        df = self.__data.copy()
        missing_and_zero = df[(df["domain"].isnull()) & (df["event_type"] == 1)]
        
        print("Missing values with zero (0) event")
        print(f"{len(missing_and_zero)} out of {len(df[df["domain"].isnull()])} missing values correspond to the dominant class 0")
        print("-" * 40)

    def __drop_missing_values(self) -> None:
        """ 
        From the training/test set - Drop all missing values using the following logic.
            Only the domain feature contains missing values
            and the vast majority of them correspond to the dominant class (event_type 0).
            In addition, the domain missing values represent the almost the 13% of the feature.
        
        From the unseen test data - Drop all missing values - No further analysis is performed.
        """
        # Data used for training and test
        self.__data.dropna(inplace=True)
        
        # Unseedn data for final test
        self.__unseen_data.dropna(inplace=True)

    def __analyse_duplicated_rows(self) -> None:
        """ 
        Count duplicated rows.
        Calculate the representation of duplicated rows on the dominant class
        """
        duplicated_rows = self.__data[self.__data.duplicated()]
        dupl_and_zero = duplicated_rows[duplicated_rows["event_type"] == 0]

        print("Duplicates rows & Duplicated rows with zero (0) event")
        print(f"Duplicated rows count: {len(duplicated_rows)}")
        print(f"{len(dupl_and_zero)} out of {len(duplicated_rows)} duplicated rows correspond to the dominant class 0")
        print("-" * 40)

    def __remove_duplicated_rows(self) -> None:
        """ 
        Remove all duplicated rows using the followng logic.
        All duplicated rows represent 0 events (the dominant class).
        """
        self.__data.drop_duplicates(inplace=True)

    def __calculate_cramers_v(self, var1: pd.Series, var2: pd.Series):
        """ Calculate Cramer's V. """
        # Create a contingency table
        contingency_table = pd.crosstab(var1, var2)

        # Perform chi-squared test on the contingency table
        chi2, p_val, dof, expected = chi2_contingency(contingency_table)

        # Calculate Cramer's V
        n = contingency_table.sum().sum()
        cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))

        return float(cramers_v)

    def __calc_categorical_feature_correlation(self):
        """
        Extract only catgorical features from the data.
        Prepare matrix.
        Create 
        """
        # Prepare an empty matrix
        matrix = pd.DataFrame(
            index=self.__categorical_features,
            columns=self.__categorical_features
        )

        # Calculate correlation for each pair of features
        for col1 in self.__categorical_features:
            for col2 in self.__categorical_features:

                # Fill in the matrix with Cramer's V
                matrix.loc[col1, col2] = self.__calculate_cramers_v(
                    self.__data[col1],
                    self.__data[col2]
                )

        # Convert matrix data into numeric
        matrix = matrix.apply(pd.to_numeric, errors='coerce')

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

        # plt.savefig('correlation.png', transparent=True)

        if self.__config.generic_config.show_plots:
                plt.show()
