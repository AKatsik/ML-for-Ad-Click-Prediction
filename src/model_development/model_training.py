""" Classification model training """

import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
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
                 processed_train_data: pd.DataFrame) -> None:
        self.__config: ConfigLoader = config
        self.__training_data: pd.DataFrame = processed_train_data

        self.__x_train: pd.DataFrame
        self.__x_test: pd.DataFrame
        self.__y_train: pd.DataFrame
        self.__y_test: pd.DataFrame

        self.__split_train_test_data()
        # self.__group_categories_by_frequency()
        # self.__group_device_type()
        # self.__encode_string_to_numeric()

        # self.__class_weights: dict = self.__calculate_class_weights()
        # self.__model = self.__init_classification_model()
        # self.__random_search = self.__prepare_hyper_param_tuning()
        # self.__execute_training()

        # self.__y_pred, self.__y_prob = self.predict(self.__x_test)
        # self.__calculate_metrics()

    @property
    def config(self) -> ConfigLoader:
        """ Config attribute. """
        return copy.deepcopy(self.__config)

    @property
    def training_data(self) -> pd.DataFrame:
        """ Data attribute. """
        return self.__training_data.copy()

    def __split_train_test_data(self):
        """ 
        Split data into training and testing sets.
        Stratify setting is activated because to make sure that the target labels 
        are equally splitted in both sets.
        """

        # Splitting features and target
        x: pd.DataFrame = self.__training_data.drop(columns=["event_type"])  # Features
        y: pd.Series = self.__training_data["event_type"]  # Target

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

    def __group_categories_by_frequency(self):
        """ Group categories in categorical variables using the frequency analysis. """

        # Settings for transformation:
        # The key shows the columnn name
        # The first element of the tuple shows the number of the most frequent values that is retained
        settings = {
            "browser": (3, 1),
            "carrier": (1, 1),
            "language": (1, 1),
            "region": (1, 1),
            "os_extended": (9, 1),
            "user_hour": (14, 1)
        }

        for item, (x1, x2) in settings.items():
            # Identify the most frequent values in the training set
            top_values = self.__x_train[item].value_counts().nlargest(x1).index.tolist()

            # Replace values in the training set not in the top 4 with "other"
            self.__x_train[item] = self.__x_train[item].apply(lambda x: x if x in top_values else 'other')

            # Apply the same transformation to the test set, keeping the same top values
            self.__x_test[item] = self.__x_test[item].apply(lambda x: x if x in top_values else 'other')

            # Apply the same transformation to the unseen data
            self.__unseen_data[item] = self.__unseen_data[item].apply(lambda x: x if x in top_values else 'other')

    def __group_device_type(self) -> None:
        """
        Group the "device_type" feature based on the following logic:
            pc & other devices is the most frequent value and it is retained.
            table and phone values are combined into mobile devices
            The rest are grouped as tv & other
        """
        # Logic of groupping device type
        def categorize_device(device):
            if device in ['phone', 'tablet']:
                return 'mobile devices'
            elif device in ['pc', 'other']:
                return device
            else:
                return 'tv and other'

        # Apply the logic
        self.__x_train['device_type'] = self.__x_train['device_type'].apply(categorize_device)
        self.__x_test['device_type'] = self.__x_test['device_type'].apply(categorize_device)
        self.__unseen_data['device_type'] = self.__unseen_data['device_type'].apply(categorize_device)

    def __encode_string_to_numeric(self) -> None:
        """ Encode string values to numeric using the model that fits only the training set. """
        for col in self.unseen_data.columns:
            if self.unseen_data[col].dtype == 'object':
                label_encoder = LabelEncoder()
                # Fit and transform the training set 
                self.__x_train[col] = label_encoder.fit_transform(self.__x_train[col].astype(str))
                
                # Transform the test set
                self.__x_test[col] = label_encoder.fit_transform(self.__x_test[col].astype(str))
                
                # Transform the unseen data
                self.__unseen_data[col] = label_encoder.fit_transform(self.__unseen_data[col].astype(str))

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

    def __init_classification_model(self):
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

    def predict(self, test_data:pd.DataFrame):
        """ Predict binary class click/no click and class probabilities using best model. """
        y_pred = self.__random_search.best_estimator_.predict(test_data)
        y_prob = self.__random_search.best_estimator_.predict_proba(test_data)[:, 1]

        return y_pred, y_prob

    def __calculate_metrics(self):
        report = classification_report(self.__y_test, self.__y_pred)
        print(report)

        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(
            self.__y_test,
            self.__y_prob
        )

        # Calculate PR AUC
        pr_auc = average_precision_score(
            self.__y_test,
            self.__y_prob
        )

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='best')
        plt.grid()
        # plt.savefig('PR_curve.png', transparent=True)

        if self.__config.generic_config.show_plots:
            plt.show()
