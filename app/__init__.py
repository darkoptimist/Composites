from flask import Flask
from keras.models import load_model
import numpy as np
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)
matModel = load_model('models/matmod')
pprModel = pickle.load(open("models/ppr/model_ppr.pickle", "rb"))
from app import routes
