import sys
import os
import pandas as pd
from src.pipeline.exception import CustomException
from src.pipeline.logger import logging
from src.pipeline.utils import load_object

# Prediction Pipeline
class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path = os.path.join('artifacts', 'model.pkl')

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            data_scaled = preprocessor.transform(features)

            pred = model.predict(data_scaled)
            return pred

        except Exception as e:
            logging.info("Exception occurred in prediction")
            raise CustomException(e, sys)  # type: ignore


# CustomData class for Algerian Forest Fire Prediction
class CustomData:
    def __init__(self, Temperature, RH, Ws, Rain, FFMC, DMC, DC, ISI, BUI, FWI, Classes, Region):
        self.Temperature = Temperature
        self.RH = RH
        self.Ws = Ws
        self.Rain = Rain
        self.FFMC = FFMC
        self.DMC = DMC
        self.DC = DC
        self.ISI = ISI
        self.BUI = BUI
        self.FWI = FWI
        self.Classes = Classes
        self.Region = Region

    def get_data_as_dataframe(self):
        try:
            data_dict = {
                "Temperature": [self.Temperature],
                "RH": [self.RH],
                "Ws": [self.Ws],
                "Rain": [self.Rain],
                "FFMC": [self.FFMC],
                "DMC": [self.DMC],
                "DC": [self.DC],
                "ISI": [self.ISI],
                "BUI": [self.BUI],
                "FWI": [self.FWI], 
                "Classes": [self.Classes],
                "Region": [self.Region]
            }
            return pd.DataFrame(data_dict)
        except Exception as e:
            raise CustomException(e, sys) # type: ignore