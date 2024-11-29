# Import libraries
import librosa
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
# from tensorflow.keras.utils import to_categorical

import os

# Load CREMA-D Dataset
paths = []
labels = []

for dirname, _, filenames in os.walk(r'G:\My Drive\Databases_FYP\CREMA-D'):
    for filename in filenames:
        if filename.endswith('.wav'):
            paths.append(os.path.join(dirname, filename))
            emotion = filename.split('_')[2]  # Get the emotion code (e.g., 'ANG')
            labels.append(emotion.lower())

print(paths[:5])

print(labels[:5])

# Create DataFrame
df = pd.DataFrame()
df['speech'] = paths
df['label'] = labels

# Map emotion codes to full labels
emotion_map = {
    'ang': 'anger',
    'dis': 'disgust',
    'fea': 'fear',
    'hap': 'happy',
    'neu': 'neutral',
    'sad': 'sad'
}

df['label'] = df['label'].map(emotion_map)

print(df.head())  # Check the first few rows

print(df['label'].value_counts()) # Check number of labels