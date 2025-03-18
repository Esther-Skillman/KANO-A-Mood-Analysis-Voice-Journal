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
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from collections import Counter
from fastapi.middleware.cors import CORSMiddleware

scaler = StandardScaler()

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or replace with specific origins like ['http://localhost:8000']
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Load CNN model
MODEL_PATH = r"C:\Repos\KANO-A-Mood-Analysis-Voice-Journal\KANO\Models\93_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Suprise']

# Audio processing functions
def extract_features(audio, sr, n_mels=128, n_fft=2048, hop_length=512,  max_shape=(128, 128)):
    #mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmax=8000)
    #log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    #log_mel_spec = librosa.util.fix_length(log_mel_spec, size=128, axis=0)
    #log_mel_spec = librosa.util.fix_length(log_mel_spec, size=128, axis=1)

    #data, sr =librosa.load(path)

    # aud=mel_spectogram(audio, sr)

    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048, hop_length=512, n_mels=128, fmax=8000)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max) # =np.max normalised relative to the most intense value
    log_mel_spec = log_mel_spec[..., np.newaxis]

    feature = librosa.util.fix_length(log_mel_spec, size=max_shape[0], axis=0)
    feature = librosa.util.fix_length(log_mel_spec, size=max_shape[1], axis=1)

    # Reshape the feature to 2D (flatten it), standardise it, and then reshape it back
    feature_reshaped = feature.reshape(-1, feature.shape[-1])
    feature_standardized = scaler.fit_transform(feature_reshaped)
    
    feature_standardized = feature_standardized.reshape(feature.shape)  # Reshape it back to original shape

    # X.append(feature_standardized)

    return feature_standardized

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

    segment_lengths = [x * 0.25 for x in range(8, 21)]  # 2 to 5 in 0.25 second increments

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

    predictions = predict_emotion(audio_data, sr, 2.5)
    percentages = calculate_emotion_percentages(predictions)

    return {"emotion_percentages": percentages}


def predict_emotion(audio_data, sr, segment_length):
    segment_samples = int(segment_length * sr)
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

    for start in range(0, len(audio_data), segment_samples):
        segment = audio_data[start:start + segment_samples]
        if len(segment) < segment_samples:
            break 
        
        features = extract_features(segment, sr)
        features = np.expand_dims(features, axis=0)

        pred = model.predict(features)[0]
        predicted_label = np.argmax(pred)
        predictions.append(emotion_labels[predicted_label])

    return predictions

@app.post("/predit_timeline/")
async def predict_timeline(file: UploadFile = File(...)):
    audio_data, sr = librosa.load(file.file, sr=None)

    segment_samples = int(2.5 * sr)
    timestamps = []
    emotions = []

    for start in range(0, len(audio_data), segment_samples):
        segment = audio_data[start:start + segment_samples]
        if len(segment) < segment_samples:
            break
        
        features = extract_features(segment, sr)
        features = np.expand_dims(features, axis=0)

        pred = model.predict(features)[0]
        predicted_label = np.argmax(pred)
        predicted_emotion = emotion_labels[predicted_label]

        timestamps.append(start / sr)
        emotions.append(predicted_emotion)

    fig, ax = plt.subplots(figsize=(10,6))

    emotion_mapping = {label: i for i, label in enumerate(emotion_labels)}
    emotion_numeric = [emotion_mapping[emo] for emo in emotions]

    ax.plot(timestamps, emotion_numeric, marker='o', linestyle='-', color='b', alpha=0.7)
    ax.set_yticks(list(emotion_mapping.values()))
    ax.set_yticklabels(emotion_labels)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Predicted Emotion')
    ax.set_title('Emotion Timeline Over Audio Duration')

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")
    