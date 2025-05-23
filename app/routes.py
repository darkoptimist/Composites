from app import app, matModel, pprModel
from flask import render_template, request
import numpy as np
from sklearn.ensemble import RandomForestRegressor

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    print(list(request.form.values()))
    features = np.array([float(i) for i in request.form.values()])
    predictmat = round(matModel.predict(features.reshape(1,-1))[0][0], 2)
    predictppr = pprModel.predict(features.reshape(1,-1))[0]
    return render_template('index.html',
                           prediction_text="Рекомендуемое соотношение матрица-наполнитель: {:.2f}\n" \
                                            "Прочность при растяжении: {:.2f}".format(predictmat, predictppr))
