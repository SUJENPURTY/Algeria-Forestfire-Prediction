import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from src.pipeline.exception import CustomException
from src.pipeline.logger import logging

## 1. Save object to file
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys) # type: ignore

## 2. Evaluate classification models
def evaluate_classification_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report[name] = accuracy
        return report
    except Exception as e:
        logging.info('Exception occurred during model evaluation')
        raise CustomException(e, sys) # type: ignore

## 3. Load object (e.g., model or preprocessor)
def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception occurred in load_object function')
        raise CustomException(e, sys) # type: ignore
