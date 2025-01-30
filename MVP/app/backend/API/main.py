# pip install flask

from flask import Flask, request, jsonify
import librosa
import numpy as np
import tensorflow as tf
import os
import tempfile
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load CNN model
MODEL_PATH = "cnn_model_test.keras"
model = tf.keras.models.load_model(MODEL_PATH)

