from app import app, pprModel
from flask import render_template, request
import numpy as np
from sklearn.ensemble import AdaBoostRegressor

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = np.array([float(i) for i in request.form.values()])
    predictppr = pprModel.predict(features.reshape(1,-1))[0]
    return render_template('index.html',
                           prediction_text="Прочность при растяжении: {:.2f}".format(predictppr))
