from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

from PIL import Image
import cv2
import numpy as np


# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH2 = 'D:/isha_model/finalmodel-vgg16 (8).hdf5'

# Load your trained model
model = load_model(MODEL_PATH2)


def model_predict(img_path, model):
    img = cv2.imread(img_path)

    # Preprocess the image
    img = cv2.resize(img, (160, 160)) / 255.
   # Reshape the image to match the input shape of the model
    img = np.expand_dims(img, axis=0)
    # Make the prediction
    preds = model.predict(img)[0]
    return preds


@app.route('/Xray', methods=['GET'])
def index():
    # Main page
    return render_template('index1.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
         basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        
        # Process prediction result
        if preds < 0.5:
            result = 'Normal'
        else:
            result = 'Pneumonia'

        print(np.max(preds))
        percentage = (np.max(preds)) 
        data = {"result": result, "percentage": percentage}
    return data

    return None


if __name__ == '__main__':
    app.run(debug=True,port=8001)

