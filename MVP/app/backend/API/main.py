# pip install flask

from flask import Flask, request, jsonify
import librosa
import numpy as np
import tensorflow as tf
import os
import tempfile
from sklearn.preprocessing import MinMaxScaler
from werkzeug.utils import secure_filename

# Load CNN model
MODEL_PATH = "cnn_model_test.keras"
model = tf.keras.models.load_model(MODEL_PATH)

def extract_mfcc(file_path, n_mfcc=40, max_pad_len=256):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)

        if mfcc.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0,0), (0, pad_width)), mode = 'constant')
        else:
            mfcc = mfcc[:, :max_pad_len]

            scaler = MinMaxScaler(feature_range=(0, 1))
            mfcc_normalised = scaler.fit_transform(mfcc.T).T 

            return mfcc_normalised
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None
    
def split_audio_to_chunks(file_path, chunk_duration=5): # In seconds
    audio, sample_rate = librosa.load(file_path, sr=None)

    total_duration = librosa.get_duration(y=audio, sr=sample_rate)
    chunks = []

    for start in np.arrange(0, total_duration, chunk_duration): # (start, stop, step)
        end = min(start + chunk_duration, total_duration)
        chunk = audio[int(start * sample_rate):int(end * sample_rate)]
        chunks.append(chunk)

    return chunks

def predict_emotion_for_chunks(chunks):
    predictions = []

    for chunk in chunks:
        mfcc = extract_mfcc(chunk)

        if mfcc is not None:
            mfcc = np.expand_dims(mfcc, axis=0)
            mfcc = np.expand_dims(mfcc, axis=-1)
            predictions = model.predict(mfcc)

            emotion = np.argmax(predictions)
            predictions.append(emotion)
            
    return predictions
