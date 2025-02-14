# Import libraries
# KERNEL SETUP IN VS CODE:
# conda create -n myenv python=3.12.2
# conda activate myenv

# %pip install resampy tf_keras tensorflow librosa pandas matplotlib kagglehub seaborn

import IPython.display as ipd
from IPython.display import Audio
import kagglehub
import librosa
from librosa import feature
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os

# Datasets via kagglehub

cremad = kagglehub.dataset_download("ejlok1/cremad")
print("CREMA-D to dataset files:", cremad)

ravdess = kagglehub.dataset_download("uwrfkaggler/ravdess-emotional-speech-audio")
print("RAVDESS to dataset files:", ravdess)

tess = kagglehub.dataset_download("ejlok1/toronto-emotional-speech-set-tess")
print("RAVDESS to dataset files:", tess)

savee = kagglehub.dataset_download("ejlok1/surrey-audiovisual-expressed-emotion-savee")
print("RAVDESS to dataset files:", savee)

# Load CREMA-D Dataset
paths = []
emotions = []

for dirname, _, filenames in os.walk(cremad): # (dirname, subdirs, filenames)
    for filename in filenames:
        if filename.endswith('.wav'):
            paths.append(os.path.join(dirname, filename))
            emotion = filename.split('_')[2]  # Get the emotion code (e.g., 'ANG')
            emotions.append(emotion)

print(paths[:5])

print(emotions[:5])

# Create DataFrame
cremad_df = pd.DataFrame()
cremad_df['paths'] = paths
cremad_df['emotions'] = emotions

# Map emotion codes to full emotions
emotion_map = {
    'ANG': 'anger',
    'DIS': 'disgust',
    'FEA': 'fear',
    'HAP': 'happy',
    'NEU': 'neutral',
    'SAD': 'sad'
}

cremad_df['emotions'] = cremad_df['emotions'].map(emotion_map)

print(cremad_df.head())

print(cremad_df['emotions'].value_counts())

paths = []
emotions = []

for dirname, _, filenames in os.walk(ravdess): # (dirname, subdirs, filenames)
    for filename in filenames:
        if filename.endswith('.wav'):
            paths.append(os.path.join(dirname, filename))
            part = filename.split('.')[0].split('-')  # Get the emotion number (e.g., '03' = happy)
            emotions.append(int(part[2]))


print(paths[:5])
print(emotions[:5])

# Create DataFrame
ravdess_df = pd.DataFrame()
ravdess_df['paths'] = paths
ravdess_df['emotions'] = emotions


# Map emotion codes to full emotions
emotion_map = {
    1 : 'neutral',
    2 : 'neutral', # calm as neutral to balance dataset
    3 : 'happy',
    4 : 'sad',
    5 : 'anger',
    6 : 'fear',
    7 : 'disgust',
    8 : 'suprise'

}

ravdess_df['emotions'] = ravdess_df['emotions'].map(emotion_map)

print(ravdess_df.head())

print(ravdess_df['emotions'].value_counts())

paths = []
emotions = []

for dirname, _, filenames in os.walk(tess): # (dirname, subdirs, filenames)
    for filename in filenames:
        if filename.endswith('.wav'):
            paths.append(os.path.join(dirname, filename))
            emotion = filename.split('.')[0].split('_')[2]  # Get the emotion code (e.g., 'ANG')
            emotions.append(emotion)

print(paths[:5])

print(emotions[:5])

# Create DataFrame
tess_df = pd.DataFrame()
tess_df['paths'] = paths
tess_df['emotions'] = emotions

# Map emotion codes to full emotions
emotion_map = {
    'angry': 'anger',
    'disgust': 'disgust',
    'fear': 'fear',
    'happy': 'happy',
    'neutral': 'neutral',
    'ps' : 'suprise',
    'sad' : 'sad'
}

tess_df['emotions'] = tess_df['emotions'].map(emotion_map)

print(tess_df.head())

print(tess_df['emotions'].value_counts())

paths = []
emotions = []

for dirname, _, filenames in os.walk(savee): # (dirname, subdirs, filenames)
    for filename in filenames:
        if filename.endswith('.wav'):
            paths.append(os.path.join(dirname, filename))
            part = filename.split('_')[1]  # Get the emotion code (e.g., 'ANG')
            emotion = part[:-6]
            emotions.append(emotion)

print(paths[:5])

print(emotions[:5])

# Create DataFrame
savee_df = pd.DataFrame()
savee_df['paths'] = paths
savee_df['emotions'] = emotions

emotion_map = {
    'n': 'neutral',
    'd': 'disgust',
    'a': 'anger',
    'f': 'fear',
    'h': 'happy',
    'sa': 'sad',
    'su' : 'suprise'
}

savee_df['emotions'] = savee_df['emotions'].map(emotion_map)

print(savee_df.head())

print(savee_df['emotions'].value_counts())

emotion_data = pd.concat([cremad_df, ravdess_df, tess_df, savee_df], axis = 0)

emotion_data.to_csv("emotion_data.csv", index=False)

print(emotion_data.emotions.value_counts())

import matplotlib.pyplot as plt
import seaborn as sns

plt.title('Emotions Count', size=16)
sns.countplot(emotion_data.emotions)
plt.xlabel('Count', size=12)
plt.ylabel('Emotions', size=12)
sns.despine(top=True, right=True, left=False, bottom=False)
plt.show()

data, sr = librosa.load(paths[0], sr=None) #Latest path value from SAVEE (angry)
print(emotions[0])
ipd.Audio(data,rate=sr)

n_mels = 128
n_fft = 2048
hop_length = 512
fmax = 8000
mel_spectogram = librosa.feature.melspectrogram(y=data, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmax=fmax)

log_mel_spectogram = librosa.power_to_db(mel_spectogram)

plt.figure(figsize=(10, 6))
librosa.display.specshow(log_mel_spectogram, x_axis='time', y_axis='mel', sr=sr)
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectogram')
plt.show()

mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=30)

plt.figure(figsize=(16, 10))
plt.subplot(3,1,1)
librosa.display.specshow(mfcc, x_axis='time')
plt.ylabel('MFCC')
plt.colorbar()

# Time Stretching
def stretch(audio, rate=0.8):
    return librosa.effects.time_stretch(audio, rate=0.8)

# Pitch Shifting
def pitch(audio, sr):
    return librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=2)

# Add Noise
def noise(audio, noise_level=0.005):
    noise_amp = noise_level * np.amax(data)
    audio = audio + noise_amp * np.random.normal(0, 1, len(data))
    return audio

# Shifting (Time warping)
def shift(audio):
    return np.roll(audio, shift=int(sr * 0.2))

n_mels = 128
n_fft = 2048
hop_length = 512
fmax = 8000

features = []

def extract_features(path, sr=None):

    audio, sr = librosa.load(path, sr=sr)

    augmented_audio = [ audio, shift(audio), noise(audio), pitch(audio, sr), stretch(audio) ]
    
    max_time_steps = 0
    
    for data in augmented_audio:

        # Find max feature time step
        for data in augmented_audio:
            mel_spectogram = librosa.feature.melspectrogram(y=data, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmax=fmax)
            log_mel_spectogram = librosa.power_to_db(mel_spectogram)
            max_time_steps = max(max_time_steps, log_mel_spectogram.shape[1])

        # MFCC
        # mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=30)

        # time_steps = min(log_mel_spectogram.shape[1], mfcc.shape[1])
        # log_mel_spectogram = log_mel_spectogram[:, :time_steps]
        # mfcc = mfcc[:, :time_steps]

        for data in augmented_audio:
            mel_spectogram = librosa.feature.melspectrogram(y=data, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmax=fmax)
            log_mel_spectogram = librosa.power_to_db(mel_spectogram)

            if log_mel_spectogram.shape[1] < max_time_steps:
                pad_width = max_time_steps - log_mel_spectogram.shape[1]
                log_mel_spectogram = np.pad(log_mel_spectogram, ((0,0), (0, pad_width)), mode='constant')
            elif log_mel_spectogram.shape[1] > max_time_steps:
                log_mel_spectogram = log_mel_spectogram[:, :max_time_steps]

        features.append(log_mel_spectogram)

    np_features = np.array(features)

    return np_features

np_features = extract_features(paths[0])
print(np_features.shape)

import multiprocessing as mp
print("Number of processors: ", mp.cpu_count())