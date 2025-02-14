# pip install fastapi
# uvicorn FastAPI:app --reload
# uvicorn FastAPI:app --host 0.0.0.0 --port 8000

# pip install flask python-multipart
import uvicorn
# from flask import Flask, request, jsonify
import librosa
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import tempfile
from sklearn.preprocessing import MinMaxScaler
# from werkzeug.utils import secure_filename
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse
import io
import matplotlib.pyplot as plt
# from main import split_audio_to_chunks, predict_emotion_for_chunks
from sklearn.preprocessing import LabelEncoder, StandardScaler
from collections import Counter

scaler = StandardScaler()

app = FastAPI()

# Load CNN model
MODEL_PATH = r"C:\KANO-A-Mood-Analysis-Voice-Journal\KANO\Models\93_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

emotion_labels = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'suprise']

# Audio processing functions
def extract_features(audio, sr, n_mels=128, n_fft=2048, hop_length=512):
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmax=8000)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    log_mel_spec = librosa.util.fix_length(log_mel_spec, size=128, axis=0) 
    log_mel_spec = librosa.util.fix_length(log_mel_spec, size=128, axis=1)
    return log_mel_spec[..., np.newaxis]

def calculate_emotion_percentages(predictions):
    counter = Counter(predictions)
    total = sum(counter.values())
    percentages = {emotion: round((count / total) * 100, 2) for emotion, count in counter.items()}
    return percentages

@app.post("/predict_graph/")
async def predict(file: UploadFile = File(...)):
    audio_data, sr = librosa.load(file.file, sr=None)

    audio_length = len(audio_data) / sr
    print(f"Audio length: {audio_length} seconds")

    segment_lengths = [x * 0.25 for x in range(4, 21)]  # 1 to 5 in 0.25 second increments

    emotion_results = {}

    for segment_length in segment_lengths:
        print(f"Processing segment length: {segment_length} seconds")

        predictions = predict_emotion(audio_data, sr, segment_length)
        percentages = calculate_emotion_percentages(predictions)
        emotion_results[segment_length] = percentages

    fig, ax = plt.subplots(figsize=(10, 6))

    for emotion in emotion_labels:
        emotion_percentages = [emotion_results[segment_length].get(emotion, 0) for segment_length in segment_lengths]
        ax.plot(segment_lengths, emotion_percentages, label=emotion)

    ax.set_xlabel('Segment Length (seconds)')
    ax.set_ylabel('Emotion Percentage (%)')
    ax.set_title('Emotion Distribution Across Different Segment Lengths')
    ax.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")

@app.post("/predict_simple/")
async def predict(file: UploadFile = File(...)):
    audio_data, sr = librosa.load(file.file, sr=None)  # Load audio

    predictions = predict_emotion(audio_data, sr)
    percentages = calculate_emotion_percentages(predictions)

    return {"emotion_percentages": percentages}

def predict_emotion(audio_data, sr):
    segment_length = 2.5  # seconds
    step = 2 # seconds (overlapping)
    segment_samples = int(segment_length * sr)
    step_samples = int(step * sr)

    predictions = []

    # If audio is shorter than segment length, pad the audio
    if len(audio_data) < segment_samples:
        padding = segment_samples - len(audio_data)
        audio_data = np.pad(audio_data, (0, padding), mode='constant')
        features = extract_features(audio_data, sr)
        features = np.expand_dims(features, axis=0)
        pred = model.predict(features)[0]
        predicted_label = np.argmax(pred)
        return [emotion_labels[predicted_label]]

    # Otherwise, process segments normally
    for start in range(0, len(audio_data) - segment_samples, step_samples):
        segment = audio_data[start:start + segment_samples]
        features = extract_features(segment, sr)
        features = np.expand_dims(features, axis=0)

        pred = model.predict(features)[0]
        predicted_label = np.argmax(pred)
        predictions.append(emotion_labels[predicted_label])

    return predictions
