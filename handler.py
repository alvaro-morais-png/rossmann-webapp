import os
import pickle 
import pandas as pd
import requests
from flask import Flask, Response, request
from rossmann.Rossmann import Rossmann

#loading model
model = pickle.load( open('model/model_rossman.pkl', 'rb'))

#initialize API
app = Flask(__name__)

@app.route('/rossmann/predict', methods=['POST'])
def rossmann_predict():
    test_json = request.get_json()

    if test_json:
        try:
            if isinstance(test_json, dict):
                test_raw = pd.DataFrame(test_json, index=[0])
            else:
                test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())

            pipeline = Rossmann()
            df1 = pipeline.data_cleaning(test_raw)
            df2 = pipeline.feature_engineering(df1)
            df3 = pipeline.data_preparation(df2)
            df4 = pipeline.get_prediction(model, test_raw, df3)

            return df4

        except Exception as e:
            return Response(
                response=str(e),
                status=500,
                mimetype='application/json'
            )
    else:
        return Response('{}', status=200, mimetype='application/json')

if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    app.run( '0.0.0.0', port=port )    