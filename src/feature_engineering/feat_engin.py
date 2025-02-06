""" Feature Engineering for training and infrence data. """

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from ..config.config_loading import ConfigLoader

def encode_data(
    data: pd.DataFrame,
    encoders: dict | None = None) -> None:
    """ 
    Encode string values to numeric.
    Also return the encoding models to be used for transformation 
    of test and inference data.
    """
    # If encoder is NOT given
    if encoders is None:
        encoders = {}
        for col in data.columns:
            if data[col].dtype == 'object':
                label_encoder = LabelEncoder()
                # Fit and transform data
                data[col] = label_encoder.fit_transform(
                    data[col].astype(str)
                )
                encoders[col] = label_encoder
    else:
        for col in data.columns:
            if data[col].dtype == 'object':
                label_encoder = encoders[col]
                # Use given encoder to transform data
                data[col] = label_encoder.transform(
                    data[col].astype(str)
                )
    return data, encoders


class FeatureEngineer:
    """ 
    Class responsible for feature engineering.
    Apply the following:
        - Remove unused data features
        - Group rare categories uning the frequency analysis
    
    """

    def __init__(self,
                 config: ConfigLoader,
                 data: pd.DataFrame,
                #  frequency_analysis_obj: FrequencyAnalyser | None = None
                ) -> None:
        self.__config: ConfigLoader = config
        self.__data: pd.DataFrame = data
        self.__x_train: pd.DataFrame
        self.__x_test: pd.DataFrame
        self.__y_train: pd.DataFrame
        self.__y_test: pd.DataFrame

        # --- Remove unused data features
        self.__remove_unused_features()

    @property
    def engineered_data(self):
        """ Engineered data property. """
        return self.__data.copy()

    @property
    def training_predictors(self) -> pd.DataFrame:
        """ Training predictors property. """
        return self.__x_train.copy()

    @property
    def test_predictors(self) -> pd.DataFrame:
        """ Test predictors property. """
        return self.__x_test.copy()

    @property
    def training_targets(self) -> pd.DataFrame:
        """ Training targets property. """
        return self.__y_train.copy()

    @property
    def test_targets(self) -> pd.DataFrame:
        """ Test targets property. """
        return self.__y_test

    def __remove_unused_features(self) -> None:
        """ 
        Remove unused features from given data based on the following logic:
            The correlation analysis reveales highly correlated features.
            Only one of these features is retained.
            From the highly correlated features, those with more categories are removed.
            
            We keep the features with less categories because we assume that they provides a
            better overview of the data.
        """

        # Remove from training data
        self.__data.drop(
            columns=self.__config.ml_config.features_removed,
            inplace=True
        )

    def group_rare_categories(
        self,
        categorical_features: list,
        frequency_analysis: dict  
    ) -> None:
        """
        Group rare categories in a category called 'Other'.
        The dominant categories are retained. 
        The rarity of categories is calculated based on the predefined threshold 
        regarding data representation.
        """
        threshold = self.__config.generic_config.grouping_threshold

        for col in self.__data:
            if col in categorical_features and "event" not in col:
                # Cumulative sum of frequency
                cumsum = frequency_analysis[col].cumsum()

                # Ensure at least one category is retained
                retained_categories = frequency_analysis[col][
                    (cumsum <= threshold) | (cumsum == cumsum.min())
                ].index

                self.__data[col] = self.__data[col].apply(
                    lambda x: x if x in retained_categories else "Other"
                )

    def split_train_test_data(self):
        """ 
        Split data into training and testing sets.
        Stratify setting is activated because to make sure that the target labels 
        are equally splitted in both sets.
        """

        # Splitting features and target
        x: pd.DataFrame = self.__data.drop(columns=["event_type"])  # Features
        y: pd.Series = self.__data["event_type"]  # Target

        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=self.__config.ml_config.test_size,
            random_state=self.__config.generic_config.random_state,
            stratify=y
        )

        self.__x_train = x_train
        self.__x_test = x_test
        self.__y_train = y_train
        self.__y_test = y_test
