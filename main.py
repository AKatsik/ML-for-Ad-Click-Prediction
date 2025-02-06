"""
This is the main of the entire ML pipeline.
AdClassificationExecutor runs the entire pipeline using the given configurations.
"""

import os
import pandas as pd
from src.config.config_loading import ConfigLoader
from src.data_loading.data_loading import DataLoader
from src.data_preprocessing.data_exploration import DataExplorator, run_frequency_analysis
from src.feature_engineering.feat_engin import FeatureEngineer, encode_data
from src.model_development.model_training import ModelTrainer
from src.model_development.model_inference import Predictor
from utils.version import __version__

print(f"Running script1.py - Version: {__version__}")

if __name__ == "__main__":

    CONFIG_PATH = "config.yaml"

    # --- Load configuration ---
    config: ConfigLoader = ConfigLoader(
        config_path=CONFIG_PATH
    )

    # --- Load training data ---
    training_data = DataLoader(
        path=os.path.join(
            config.data_paths_elements.data_folder,
            config.data_paths_elements.data_file
        )
    )

    # --- Preprocess training data ---
    # Init DataExplorator
    processed_train_data: DataExplorator = DataExplorator(
        config=config,
        data=training_data.loaded_data
    )
    # Filter missing data by the binary target class
    processed_train_data.filter_missing_with_event()
    # Drop missing values - Optional decision
    processed_train_data.drop_missing_values()
    # Analyse duplicated rows - Filtered by binary target class
    processed_train_data.analyse_duplicated_rows()
    # Drop duplicated rows - Optional decision
    processed_train_data.remove_duplicated_rows()
    # Run frequency analyis for categorical features
    freq_anal_train_data: dict = run_frequency_analysis(
        data=processed_train_data.processed_data,
        categorical_features=processed_train_data.categorical_features
    )
    # Plot the frequency analysis results
    processed_train_data.plot_frequency_analysis(
        frequency_analysis=freq_anal_train_data
    )
    # Calculate relationship of categorical features using Cramer's V
    processed_train_data.calc_categorical_feature_correlation()

    # --- Feature engineering training data ---
    engineered_train_data: FeatureEngineer = FeatureEngineer(
        config=config,
        data=processed_train_data.processed_data
    )
    # Group rare categories of categorical train features into category called Other
    engineered_train_data.group_rare_categories(
        categorical_features=processed_train_data.categorical_features,
        frequency_analysis=freq_anal_train_data
    )
    # Split data into train and test
    engineered_train_data.split_train_test_data()
    x_train: pd.DataFrame = engineered_train_data.training_predictors
    y_train: pd.DataFrame = engineered_train_data.training_targets
    x_test: pd.DataFrame = engineered_train_data.test_predictors
    y_test: pd.DataFrame = engineered_train_data.test_targets

    # Encode string to numeric - training data
    encoded_x_train: pd.DataFrame
    train_encoder: dict
    encoded_x_train, train_encoder = encode_data(
        data=x_train
    )
    # Encode string to numeric - training data
    encoded_x_test: pd.DataFrame
    encoded_x_test, _ = encode_data(
        data=x_test,
        encoders=train_encoder
    )

    # --- Model training ---
    model_training = ModelTrainer(
        config=config,
        x_train=encoded_x_train,
        x_test=encoded_x_test,
        y_train=y_train,
        y_test=y_test
    )
    # Extract classification report
    classification_report: dict = model_training.classification_report
    print(classification_report)

    # --- Load inference data
    inference_data = DataLoader(
        path=os.path.join(
            config.data_paths_elements.data_folder,
            config.data_paths_elements.final_test_file
        )
    )

    # --- Preprocess inference data ---
    # Init DataExplorator
    processed_inference_data: DataExplorator = DataExplorator(
        config=config,
        data=inference_data.loaded_data
    )
    # Drop missing values - Optional decision
    processed_inference_data.drop_missing_values()

    # --- Feature engineering inference data ---
    engineered_inference_data: FeatureEngineer = FeatureEngineer(
        config=config,
        data=processed_inference_data.processed_data
    )
    # Group rare categories of categorical inference features into category called Other
    engineered_inference_data.group_rare_categories(
        categorical_features=processed_inference_data.categorical_features,
        frequency_analysis=freq_anal_train_data
    )
    # Encode string to numeric - training data
    encoded_inference_data: pd.DataFrame
    encoded_inference_data, _ = encode_data(
        data=engineered_inference_data.engineered_data,
        encoders=train_encoder
    )
    
    # --- Final prediction with inference data
    prediction = Predictor(
        classification_model=model_training.best_model,
        inference_data=encoded_inference_data
    )
