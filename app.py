from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

## Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            gender=request.form.get('gender'),
            Area=request.form.get('Area'),
            type=request.form.get('type'),
            verbal=request.form.get('verbal'),
            logical=request.form.get('logical')
        )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        # Render `pred.html` instead of `home.html`
        return render_template('pred.html', results=results[0])

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
