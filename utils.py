from keras import backend as K
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def produce_cm(y_pred, y_true):
    y_pred = np.argmax(y_pred, axis = 1)
    y_true = np.argmax(y_true, axis = 1)
    return confusion_matrix(y_true, y_pred)

def plot_cm(y_pred, y_true):
    cm = produce_cm(y_pred, y_true)
    df_cm = pd.DataFrame(cm, range(9), range(9))
    plt.figure(figsize = (8, 6))
    sn.heatmap(df_cm, annot=True, cmap="YlGnBu")
    plt.show()

"""
Code by Tasos: https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model
"""
def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1(y_true, y_pred):
    pres = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2*((pres*rec)/(pres+rec+K.epsilon()))