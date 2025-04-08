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

from joblib import Parallel, delayed
import timeit
import numpy as np
import librosa.util
from tqdm import tqdm
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical

scaler = StandardScaler()

def tqdm_parallel(iterable, total):
    for item in tqdm(iterable, total=total, desc="Processing Audio Files"):
        yield item


# Function to extract and process features
def process_feature(path, emotion, max_shape=(128, 128)):  
    features = get_features(path)  
    X = []  
    Y = []  

    for feature in features:
        # Fix length of mel
        feature = librosa.util.fix_length(feature, size=max_shape[0], axis=0)
        feature = librosa.util.fix_length(feature, size=max_shape[1], axis=1)

        # Reshape the feature to 2D (flatten it), standardise it, and then reshape it back
        feature_reshaped = feature.reshape(-1, feature.shape[-1])
        feature_standardized = scaler.fit_transform(feature_reshaped) 
        
        feature_standardized = feature_standardized.reshape(feature.shape)  # Reshape it back to original shape

        X.append(feature_standardized)  
        Y.append(emotion)  

    return X, Y

# Load data
paths = emotion_data.paths  
emotions = emotion_data.emotions  
num_files = len(paths)  

start = timeit.default_timer()

# Parallel processing with tqdm
results = Parallel(n_jobs=-1)(
    delayed(process_feature)(path, emotion) for path, emotion in tqdm_parallel(zip(paths, emotions), total=num_files)
)

# Collect results
X = []
Y = []
for result in results:
    x, y = result
    X.extend(x)
    Y.extend(y)

# Convert to NumPy arrays
X = np.stack(X, axis=0)
Y = np.array(Y)

stop = timeit.default_timer()
print(f'Time: {stop - start:.2f} seconds')

# One-Hot Encoding
label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y) 
Y_one_hot = to_categorical(Y_encoded)  # Convert to one-hot encoding

# Shape verification
assert len(Y) == X.shape[0], "Mismatch: Number of labels does not match the number of samples!"
print(f'Final X shape: {X.shape}')  # Expected (num_samples, 128, 128, 1)
print(f'Final Y shape: {Y_one_hot.shape}')  # Expected (num_samples, num_classes)

from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# Train-test split (80-20 split)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_one_hot, test_size=0.2, random_state=42, shuffle=True)

# Validation split (10% of training data)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=42)

X_train.shape, Y_train.shape, X_test.shape, Y_test.shape

# Convert to TensorFlow tensors
X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
X_val = tf.convert_to_tensor(X_val, dtype=tf.float32)
X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)

Y_train = tf.convert_to_tensor(Y_train, dtype=tf.float32)
Y_val = tf.convert_to_tensor(Y_val, dtype=tf.float32)
Y_test = tf.convert_to_tensor(Y_test, dtype=tf.float32)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout

num_classes = Y_one_hot.shape[1]  # Number of emotion categories

model = Sequential([
    # Convolutional feature extraction
    Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(128, 128, 1)),
    BatchNormalization(),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2,2)),

    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2,2)),

    # Conv2D(256, (3,3), activation='relu', padding='same'),
    # BatchNormalization(),
    # MaxPooling2D((2,2)),

    # Flatten and Fully Connected Layers
    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),

    Dense(num_classes, activation='softmax') 
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, verbose=1)

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, min_lr=1e-5, verbose=1)

# Train the CNN model
history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=50,
    batch_size=64,
    callbacks=[model_checkpoint, early_stop, lr_reduction]
)

test_loss, test_acc = model.evaluate(X_test, Y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")
print(f"Test Lost: {test_loss:.4f}")


stopped_epoch = early_stop.stopped_epoch

epochs = [i for i in range(stopped_epoch+1)]
fig , ax = plt.subplots(1,2)
train_acc = history.history['accuracy']
train_loss = history.history['loss']
test_acc = history.history['val_accuracy']
test_loss = history.history['val_loss']

fig.set_size_inches(20,6)
ax[0].plot(epochs , train_loss , label = 'Training Loss')
ax[0].plot(epochs , test_loss , label = 'Testing Loss')
ax[0].set_title('Training & Testing Loss')
ax[0].legend()
ax[0].set_xlabel("Epochs")

ax[1].plot(epochs , train_acc , label = 'Training Accuracy')
ax[1].plot(epochs , test_acc , label = 'Testing Accuracy')
ax[1].set_title('Training & Testing Accuracy')
ax[1].legend()
ax[1].set_xlabel("Epochs")
plt.show()