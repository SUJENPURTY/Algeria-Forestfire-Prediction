# Basic Import
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from src.pipeline.exception import CustomException
from src.pipeline.logger import logging
from src.pipeline.utils import save_object, evaluate_classification_model  # You need a new eval function for classification

from dataclasses import dataclass
import sys
import os

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initate_model_training(self, train_array, test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Classification Models
            models = {
                'LogisticRegression': LogisticRegression(),
                'DecisionTree': DecisionTreeClassifier(),
                'RandomForest': RandomForestClassifier(),
                'GradientBoosting': GradientBoostingClassifier(),
                'SVC': SVC()
            }

            model_report: dict = evaluate_classification_model(X_train, y_train, X_test, y_test, models)
            print(model_report)
            logging.info(f'Model Report: {model_report}')
            print('\n' + '='*80 + '\n')

            # Get best model
            best_model_score = max(model_report.values())
            best_model_name = [key for key, value in model_report.items() if value == best_model_score][0]
            best_model = models[best_model_name]

            print(f'Best Model Found: {best_model_name} with Accuracy: {best_model_score}')
            logging.info(f'Best Model Found: {best_model_name} with Accuracy: {best_model_score}')
            print('\n' + '='*80 + '\n')

            # Save model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

        except Exception as e:
            logging.error('Exception occurred during Model Training')
            raise CustomException(e, sys) # type: ignore
