from flask import Flask, jsonify
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


app = Flask(__name__)

@app.route('/')
def selectmode():
   return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)