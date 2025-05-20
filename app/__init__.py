from flask import Flask
from keras.models import load_model
import numpy as np

app = Flask(__name__)
matModel = load_model('models/matmod')
print(matModel.predict(np.array([1,1,1,1,1,1,1,1,1,1]))[0][0])
from app import routes
