from app import app, matModel
from flask import render_template, request
import numpy as np

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    print(list(request.form.values()))
    features = np.array([float(i) for i in request.form.values()])


    predict = round(matModel.predict(features)[0][0], 2)
    return render_template('index.html', prediction_text="Рекомендуемое соотношение матрица-наполнитель: {:.2f}".format(predict))
