import numpy as np
from sympy import solve, symbols
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pickle
import glob
from tools import config
import cv2

def get_lambda(measures):
    g1, g2, g3 = measures
    x = symbols('x')
    lmbd = solve((g1 * g2 * g3) * x ** 3 + 
                         (g1 * g2 + g2 * g3 + g1 * g3) * x ** 2 + 
                         (g1 + g2 + g3 - 1) * x, x)
    return lmbd[2]

def sugeno_fuzzy_integral(X, lmbd):
    sorted_data = np.sort(X, order="prediction_score")
    f_prev = sorted_data[0][1]
    pred = sorted_data[0][0] * sorted_data[0][1]
    
    for i in range(1, len(sorted_data)):
        f_cur = f_prev + sorted_data[i][1] + lmbd * sorted_data[i][1] * f_prev
        pred = pred + sorted_data[i][0] * (f_cur - f_prev)
        f_prev = f_cur
    
    return pred

def ensemble(measures, *model_predictions):
    models_count = len(model_predictions)
    assert models_count == len(measures)
    
    lmbd = get_lambda(measures)
    dtype = [('prediction_score', float), ('fuzzy_measure', float)]
    final_predictions = list()
    for i in range(len(model_predictions[0])):
        score_values = [(model_predictions[j][i], measures[j]) for j in range(models_count)]
        data_belong = np.array(score_values, dtype=dtype)
        x_belong_agg = sugeno_fuzzy_integral(data_belong, lmbd)
        final_predictions.append(x_belong_agg)
   
    return np.array(final_predictions).argmax(axis=0)
