# pip install fastapi
# uvicorn FastAPI:app --reload

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
from fastapi.responses import JSONResponse
import io
import matplotlib.pyplot as plt
# from main import split_audio_to_chunks, predict_emotion_for_chunks

# Load CNN model
MODEL_PATH = r"C:\KANO-A-Mood-Analysis-Voice-Journal\MVP\app\backend\API\cnn_model_test.keras" # Replace with absoloute path
model = tf.keras.models.load_model(MODEL_PATH)

split_chunk_duration = 1 # In seconds

emotion_map = {
    0: 'anger',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad'
}

def extract_mfcc(audio, sample_rate, n_mfcc=40, max_pad_len=256):
    try:
        # audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')

        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        #print(f"MFCC Shape: {mfcc.shape}")
        #print(f"First few MFCC values: {mfcc[:, :5]}")

        if mfcc.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0,0), (0, pad_width)), mode = 'constant')
        else:
            mfcc = mfcc[:, :max_pad_len]

        scaler = MinMaxScaler(feature_range=(0, 1))
        mfcc_normalised = scaler.fit_transform(mfcc.T).T 
        #print(f"MFCC Normalised: {mfcc_normalised}")
        return mfcc_normalised
    except Exception as e:
        print(f"Error processing audio data: {e}")
        return None
    
def split_audio_to_chunks(audio, sample_rate, chunk_duration=split_chunk_duration):
    # audio, sample_rate = librosa.load(file_path, sr=None)

    total_duration = librosa.get_duration(y=audio, sr=sample_rate)
    chunks = []

    for start in np.arange(0, total_duration, chunk_duration): # (start, stop, step)
        end = min(start + chunk_duration, total_duration)
        chunk = audio[int(start * sample_rate):int(end * sample_rate)]
        chunks.append(chunk)

    return chunks

def predict_emotion_for_chunks(chunks, sample_rate):
    #print(f"Chunks and sample_rate: {chunks} + {sample_rate}")
    emotions = []
    for chunk in chunks:
        mfcc = extract_mfcc(chunk, sample_rate)
        #print(f"MFCC: {mfcc}")
        if mfcc is not None:
            mfcc = np.expand_dims(mfcc, axis=0)
            mfcc = np.expand_dims(mfcc, axis=-1)
            prediction = model.predict(mfcc)
            print(f"Single chunk predictions: {prediction}")
            emotion_index = np.argmax(prediction)
            emotion_name = emotion_map[emotion_index]
            print(f"Chunk Emotion: {emotion_name}")
            emotions.append(emotion_name)
            
    return emotions

app = FastAPI()

@app.post("/predict-emotions/")
async def predict_emotions(file: UploadFile = File(...)):
    try:
        print(f"Received file: {file.filename}")
        #file_location = f"temp_{file.filename}"
        #with open(file_location, "wb") as f:
            #f.write(await file.read())
        
        file_content = await file.read()

        audio, sample_rate = librosa.load(io.BytesIO(file_content), sr=None)
        plt.plot(audio)
        plt.title("Audio Waveform")
        plt.show()
        #print(f"Audio Extracted: {audio}")
        #print(f"Sample Rate Extracted: {sample_rate}")

        chunks = split_audio_to_chunks(audio, sample_rate)
        #print(f"Chunks list: {chunks}")
        chunk_predictions = predict_emotion_for_chunks(chunks, sample_rate)
        print(f"Overall chunk predictions: {chunk_predictions}")

        total_chunks = len(chunk_predictions)
        emotion_counts = {emotion: chunk_predictions.count(emotion) for emotion in set(chunk_predictions)}
        emotion_results = {emotion: (count / total_chunks) * 100 for emotion, count in emotion_counts.items()}

        return JSONResponse(content={"emotion_results": emotion_results})
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    
#if __name__ == "__main__":
#    uvicorn.run(app, host="127.0.0.1", port=8000)