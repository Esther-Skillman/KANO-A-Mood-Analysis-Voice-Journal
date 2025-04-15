from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import tensorflow as tf
import librosa
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import io
import os

app = FastAPI()
scaler = StandardScaler()

emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Suprise']
MODEL_PATH = os.path.join(os.path.dirname(__file__), "93_model.keras")
model = tf.keras.models.load_model(MODEL_PATH)

def extract_features(audio, sr, max_shape=(128, 128)):
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048, hop_length=512, n_mels=128, fmax=8000)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)[..., np.newaxis]
    feature = librosa.util.fix_length(log_mel_spec, size=max_shape[0], axis=0)
    feature = librosa.util.fix_length(feature, size=max_shape[1], axis=1)
    reshaped = feature.reshape(-1, feature.shape[-1])
    standardized = scaler.fit_transform(reshaped).reshape(feature.shape)
    return standardized

def predict_emotion(audio_data, sr, segment_length):
    segment_samples = int(segment_length * sr)
    predictions = []

    for start in range(0, len(audio_data), segment_samples):
        segment = audio_data[start:start + segment_samples]
        if len(segment) < segment_samples:
            break
        features = extract_features(segment, sr)
        features = np.expand_dims(features, axis=0)
        pred = model.predict(features)[0]
        predictions.append(emotion_labels[np.argmax(pred)])
    return predictions

def calculate_emotion_percentages(predictions):
    counter = Counter(predictions)
    total = sum(counter.values())
    return {emotion: round((count / total) * 100, 2) for emotion, count in counter.items()}

@app.post("/predict_simple/")
async def predict_simple(file: UploadFile = File(...)):
    audio_data, sr = librosa.load(file.file, sr=None)
    predictions = predict_emotion(audio_data, sr, 2.5)
    return {"emotion_percentages": calculate_emotion_percentages(predictions)}

# Lambda adapter
from mangum import Mangum
handler = Mangum(app)
