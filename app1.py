from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from flask import Flask, jsonify

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'D:/new pro/Deployment-Deep-Learning-Model-master/model/finalmodel-ResNet50 (2).hdf5'

# Load your trained model
model = load_model(MODEL_PATH)

         # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')


test_datagen = ImageDataGenerator(dtype='float32')
test_generator = test_datagen.flow_from_directory('E:/isha dataset/Data/test',
                                                   batch_size = 32,
                                                   target_size = (460,460),
                                                   class_mode = 'categorical')



from PIL import Image

def model_predict(img_path, model):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(460, 460))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    preds = model.predict(img_array)
    class_names = list(test_generator.class_indices.keys())
    result = class_names[np.argmax(preds)]
    percentage = round(100 * np.max(preds))
    data = {"result": result, "percentage": percentage}
    return data






@app.route('/ctscan', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


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

        print(preds)
        # Get the disease with highest probability
        result = preds
        return result
    return None



@app.route('/')
def welcome():
   return render_template('welcome.html')

@app.route('/selectmode')
def selectmode():
   return render_template('main.html')

@app.route('/about')
def about():
   return render_template('about.html')

@app.route('/service')
def service():
   return render_template('service.html')

@app.route('/contact')
def contact():
   return render_template('contact.html')







if __name__ == '__main__':
    app.run(debug=True,port=8000)
