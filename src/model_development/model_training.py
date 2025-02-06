""" Classification model training """

import os
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve, average_precision_score
from ..config.config_loading import ConfigLoader


class ModelTrainer:
    """
    Class to build and train the main ML model to predict click/no click (binary classification).
    The class also includes the necessary feature engineering.
        1. Group by feature frequency
        2. Encode strings to numeric
    """

    def __init__(self,
                 config: ConfigLoader,
                 x_train: pd.DataFrame,
                 x_test: pd.DataFrame,
                 y_train: pd.DataFrame,
                 y_test: pd.DataFrame) -> None:

        # --- Input parameters
        self.__config: ConfigLoader = config
        self.__x_train: pd.DataFrame = x_train
        self.__x_test: pd.DataFrame = x_test
        self.__y_train: pd.DataFrame = y_train
        self.__y_test: pd.DataFrame = y_test

        # --- Output
        self.__best_model: BaseEstimator
        self.__y_predictions: np.ndarray
        self.__y_probabilities: np.ndarray
        self.__classification_report: dict
        self.__feat_importance: np.ndarray

        self.__class_weights: dict = self.__calculate_class_weights()
        self.__model = self.__init_random_forest_model()
        self.__random_search = self.__prepare_hyper_param_tuning()
        self.__execute_training()

        self.__predict()
        self.__calculate_metrics()

        self.__extract_feature_importance()
        self.__plot_feature_importance()

    @property
    def class_weights(self) -> dict:
        """ Class weights property. """
        return self.__class_weights

    @property
    def best_model(self) -> BaseEstimator:
        """ Best model property. """
        return copy.deepcopy(self.__best_model)

    @property
    def predictions(self) -> np.ndarray:
        """ Predictions property. """
        return copy.deepcopy(self.__y_predictions)

    @property
    def pred_probabilities(self) -> np.ndarray:
        """ Prediction probabilities property. """
        return copy.deepcopy(self.__y_probabilities)

    @property
    def classification_report(self) -> dict:
        """ Classification report property. """
        return self.__classification_report

    def __calculate_class_weights(self) -> dict:
        """
        Calculate class weights using the frequency of the target labels.
        The weights are used in the training to penalise the dominant class.
        """

        unique_classes = np.unique(self.__y_train)

        # Compute class weights
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=unique_classes,
            y=self.__y_train
        )
        class_weight_dict = dict(zip(unique_classes, class_weights))
        return class_weight_dict

    def __init_random_forest_model(self):
        """ Initate classification model. """
        rf = RandomForestClassifier(
            class_weight=self.__class_weights, 
            random_state=self.__config.generic_config.random_state
        )
        return rf

    def __prepare_hyper_param_tuning(self):
        """
        Prepare hyper parameter optimisation.
        Random search is used to accelearte the learning process.
        Cross validation is also used to boost the learning process.
        """
        # Set hyper params
        param_dist = {
            'n_estimators': self.__config.ml_config.n_estimators,
            'max_depth': self.__config.ml_config.max_depth,
            'min_samples_split': self.__config.ml_config.min_samples_split,
            'min_samples_leaf': self.__config.ml_config.min_samples_leaf,
            'max_features': self.__config.ml_config.max_features,
            'bootstrap': self.__config.ml_config.bootstrap
        }

        random_search = RandomizedSearchCV(
            estimator=self.__model,
            param_distributions=param_dist,
            scoring='f1_macro',
            n_iter=self.__config.ml_config.iterations,
            cv=self.__config.ml_config.cv_folds,
            verbose=2,
            random_state=self.__config.generic_config.random_state,
            n_jobs=-1
        )

        return random_search

    def __execute_training(self):
        """ Execute the training process. """
        self.__random_search.fit(
            self.__x_train,
            self.__y_train
        )

    def __predict(self):
        """ 
        Extract best model from the random search.
        Predict binary class click/no click and class probabilities using best model. 
        """
        self.__best_model: BaseEstimator = self.__random_search.best_estimator_
        self.__y_predictions: np.ndarray = self.__best_model.predict(self.__x_test)
        self.__y_probabilities: np.ndarray = self.__best_model.predict_proba(self.__x_test)[:, 1]

    def __extract_feature_importance(self):
        """ Extract feature importance. """
        self.__feat_importance = self.__best_model.feature_importances_

    def __plot_feature_importance(self) -> None:
        """ Plot feature importance. """
        feature_importance_df = pd.DataFrame(
            {
                'Feature': self.__x_train.columns, 
                'Importance': self.__feat_importance
            }
        ).sort_values(by='Importance', ascending=True)

        # Plot feature importance as a bar chart
        plt.figure(figsize=(8, 6))
        plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
        plt.xlabel('Feature Importance')
        plt.ylabel('Feature')
        plt.title('Random Forest Feature Importance')

        plt.savefig(
            os.path.join(
                "plots", 
                "feature_importance.png"
            )
        )

        if self.__config.generic_config.show_plots:
            plt.show()

        plt.close()

    def __calculate_metrics(self):
        self.__classification_report: dict = classification_report(self.__y_test, self.__y_predictions)

        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(
            self.__y_test,
            self.__y_probabilities
        )

        # Calculate PR AUC
        pr_auc = average_precision_score(
            self.__y_test,
            self.__y_probabilities
        )

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='best')
        plt.grid()

        plt.savefig(
            os.path.join(
                "plots", 
                "PR_curve.png"
            ),
            transparent=True
        )

        if self.__config.generic_config.show_plots:
            plt.show()

        plt.close()
