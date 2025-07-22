import os
import sys
from flask import Flask, request, render_template
import pandas as pd

from src.pipeline.prediction_pipeline import CustomData, PredictPipeline
from src.pipeline.logger import logging
from src.pipeline.exception import CustomException

app = Flask(__name__)

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('form.html')
    else:
        try:
            # Read input from the form
            data = CustomData(
                Temperature=float(request.form.get('Temperature')), # type: ignore
                RH=float(request.form.get('RH')), # type: ignore
                Ws=float(request.form.get('Ws')), # type: ignore
                Rain=float(request.form.get('Rain')), # type: ignore
                FFMC=float(request.form.get('FFMC')), # type: ignore
                DMC=float(request.form.get('DMC')), # type: ignore
                DC=float(request.form.get('DC')), # type: ignore
                ISI=float(request.form.get('ISI')), # type: ignore
                BUI=float(request.form.get('BUI')), # type: ignore
                FWI=float(request.form.get('FWI')), # type: ignore
                Classes=float(request.form.get('Classes')), # type: ignore
                Region=int(request.form.get('Region')) # type: ignore
            )

            # Convert input data to DataFrame
            final_new_data = data.get_data_as_dataframe()
            logging.info(f"Input DataFrame:\n{final_new_data}")

            # Make prediction
            predict_pipeline = PredictPipeline()
            prediction = predict_pipeline.predict(final_new_data)

            logging.info(f"Prediction Output: {prediction}")

            # If prediction is numeric, round it. Else, display it as-is
            try:
                result = float(prediction[0])
                result = round(result, 2)
            except ValueError:
                result = prediction[0]  # It's probably a label like "Fire" or "No Fire"

            return render_template('form.html', result=result)

        except Exception as e:
            raise CustomException(e, sys) # type: ignore

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
