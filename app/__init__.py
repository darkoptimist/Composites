from flask import Flask


import numpy as np
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostRegressor

app = Flask(__name__)
pprModel = pickle.load(open("models/ppr/model_ppr.pickle", "rb"))
from app import routes
