import cv2
import tempfile
import numpy as np

from tensorflow.keras.applications.vgg16 import preprocess_input


def histogram_equalization_gray(image):
    return cv2.equalizeHist(image) 

def histogram_equalization_color(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB) 

def enhance_image(img_path):
    img = cv2.imread(img_path)  
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    
    lab_clahe = cv2.merge([l_clahe, a, b])
    
    img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    
    return img_clahe

def extract_features(img_path, model):
    img = enhance_image(img_path)  
    
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img_array = np.expand_dims(img, axis=0) 
    img_array = preprocess_input(img_array)

    features = model.predict(img_array, verbose=0)
    return features.flatten() 

def extract_features_from_uploaded_file(uploaded_file, model):
    uploaded_file.seek(0)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    return extract_features(tmp_path, model)