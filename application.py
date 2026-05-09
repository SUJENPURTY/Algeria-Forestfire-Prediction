import os
import sys
from flask import Flask, request, render_template

from src.pipeline.prediction_pipeline import CustomData, PredictPipeline
from src.pipeline.logger import logging
from src.pipeline.exception import CustomException

app = Flask(__name__)

REQUIRED_FIELDS = [
    'Temperature', 'RH', 'Ws', 'Rain',
    'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI', 'Classes', 'Region'
]

FLOAT_FIELDS = [
    'Temperature', 'RH', 'Ws', 'Rain',
    'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI', 'Classes'
]


def validate_form_inputs(form):
    errors = {}
    for field in REQUIRED_FIELDS:
        value = form.get(field)
        if not value or str(value).strip() == '':
            errors[field] = f"{field} is required"
            continue
        if field in FLOAT_FIELDS:
            try:
                float(value)
            except ValueError:
                errors[field] = f"{field} must be a valid number"
    return errors


def parse_form_values(form):
    data = {}
    for field in FLOAT_FIELDS:
        val = form.get(field)
        data[field] = val.strip() if val and val.strip() else ''
    region = form.get('Region')
    data['Region'] = region.strip() if region and region.strip() else ''
    return data


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('form.html', result=None, errors=None, values=None)

    errors = validate_form_inputs(request.form)
    if errors:
        parsed = parse_form_values(request.form)
        return render_template('form.html', result=None, errors=errors, values=parsed)

    try:
        data = CustomData(
            Temperature=float(request.form.get('Temperature')),
            RH=float(request.form.get('RH')),
            Ws=float(request.form.get('Ws')),
            Rain=float(request.form.get('Rain')),
            FFMC=float(request.form.get('FFMC')),
            DMC=float(request.form.get('DMC')),
            DC=float(request.form.get('DC')),
            ISI=float(request.form.get('ISI')),
            BUI=float(request.form.get('BUI')),
            FWI=float(request.form.get('FWI')),
            Classes=float(request.form.get('Classes')),
            Region=int(request.form.get('Region'))
        )

        final_new_data = data.get_data_as_dataframe()
        logging.info(f"Input DataFrame:\n{final_new_data}")

        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(final_new_data)

        logging.info(f"Prediction Output: {prediction}")

        if prediction is None or (hasattr(prediction, '__len__') and len(prediction) == 0):
            result = "Unable to predict"
            result_type = "general"
            logging.warning("Prediction returned empty result")
        else:
            raw = prediction[0]
            try:
                prob = float(raw)
                if 0 <= prob <= 1:
                    result = "High Fire Risk Detected" if prob > 0.5 else "Low Fire Risk"
                    result_type = "fire" if prob > 0.5 else "safe"
                else:
                    result = round(prob, 2)
                    result_type = "general"
            except (ValueError, TypeError):
                result = str(raw)
                result_type = "general"

        logging.info(f"Final Result: {result} [{result_type}]")

        parsed = parse_form_values(request.form)
        return render_template('form.html', result=result, result_type=result_type, errors=None, values=parsed)

    except CustomException as e:
        logging.error(f"CustomException during prediction: {e}")
        return render_template('form.html', result=None, result_type=None, errors={"general": "A prediction error occurred. Please check your inputs and try again."}, values=parse_form_values(request.form))

    except FileNotFoundError as e:
        logging.error(f"Model/preprocessor file not found: {e}")
        return render_template('form.html', result=None, result_type=None, errors={"general": "Model files not found. Please train the model first."}, values=None)

    except Exception as e:
        logging.error(f"Unexpected error during prediction: {e}")
        return render_template('form.html', result=None, result_type=None, errors={"general": "An unexpected error occurred. Please try again."}, values=parse_form_values(request.form))


if __name__ == "__main__":
    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    port = int(os.environ.get("FLASK_PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "True").lower() in ("true", "1", "yes")
    app.run(host=host, port=port, debug=debug)
