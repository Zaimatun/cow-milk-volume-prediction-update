import os
import joblib
import numpy as np

from tensorflow.keras.applications import VGG16

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

def load_model():
    rf_model = joblib.load(
        os.path.join(MODEL_DIR, "rf_tuned_model.joblib")
    )
    svr_model = joblib.load(
        os.path.join(MODEL_DIR, "svr_tuned_model.joblib")
    )
    xgb_model = joblib.load(
        os.path.join(MODEL_DIR, "xgb_tuned_model.joblib")
    )
    lgbm_model = joblib.load(
        os.path.join(MODEL_DIR, "lgbm_tuned_model.joblib")
    )
    
    return rf_model, svr_model, xgb_model, lgbm_model

def load_vgg16():
    vgg16_model = VGG16(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
    
    return vgg16_model

def choquet_integral(weights, inputs):
    sorted_inputs = sorted(enumerate(inputs), key=lambda x: x[1]) 
    indices, sorted_inputs = zip(*sorted_inputs)
    cumulative_weights = np.cumsum([weights[i] for i in indices])
    return np.dot(sorted_inputs, cumulative_weights)