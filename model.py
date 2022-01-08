import numpy as np
import cv2                
from keras.applications.resnet import ResNet50
from keras.preprocessing import image                  
from keras.models import Sequential
from keras.applications.resnet import preprocess_input, decode_predictions
from extract_bottleneck_features import *
import pickle
import keras

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
ResNet50_model = ResNet50(weights='imagenet')

model = keras.models.load_model("model (1)")

dog_names = open('class_names','rb')
dog_names = pickle.load(dog_names)

def face_detector(img_path):
    # img = cv2.read(img_path)
    gray = cv2.cvtColor(img_path, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


def path_to_tensor(img_path):
    x = image.smart_resize(img_path,size=(224, 224))
    # x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)

def ResNet50_predict_labels(img_path):
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))

def VGG16_predict(img_path):
    bottleneck_feature = extract_VGG16(path_to_tensor(img_path))
    predicted_vector = model.predict(bottleneck_feature)
    return dog_names[np.argmax(predicted_vector)]

def predict_dogs(img_path):
    if face_detector(img_path) or dog_detector(img_path)>0:
        return VGG16_predict(img_path).split('.')[-1]
    else:
        return 'Error neither Dog or Human was inserted'

