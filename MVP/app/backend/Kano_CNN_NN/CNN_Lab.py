# Import libraries
# pip install resampy
# pip install tensorflow
# pip install librosa
# pip install pandas
# pip install matplotlib

import librosa
from librosa import feature
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.python.keras.utils.np_utils import to_categorical

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


# Function to extract MFCCs
print("Extracting MFCCs...")
def extract_mfcc(file_path, n_mfcc=40, max_pad_len=256): # 256 * (512 / 22050) <= 6 seconds
    """
    Extract MFCC features from an audio file.
    Args:
    - file_path: Path to the audio file.
    - n_mfcc: Number of MFCCs to extract.
    - max_pad_len: Fixed length for padding/truncating.
    Returns:
    - mfcc: Numpy array of shape (n_mfcc, max_pad_len).
    """
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')

        # print("Audio Data (First 10 samples):", audio[:10])
        # print("Sample Rate:", sample_rate)  # (e.g., 22050 Hz)

        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        # print("MFCC Shape:", mfcc.shape)
        # print("MFCC (First 5 coefficients for the first few frames):")
        # print(mfcc[:, :5])

        # Pad or truncate to ensure fixed length
        if mfcc.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]
        return mfcc
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None
print("MFCCs extraction SUCCESS")

print("Loading dataset...")
# Load dataset
df['features'] = df['speech'].apply(lambda x: extract_mfcc(x))

# Drop rows with errors
df = df.dropna(subset=['features'])

print("Dataset loading SUCCESS")

print("Extracting features and labels...")

# Convert features and labels to NumPy arrays
X = np.array(df['features'].tolist())  # Features
y = pd.get_dummies(df['label']).values  # One-hot encoded labels

print("Features and labels extraction SUCCESS")

print("Fitting data for CNN model...")
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape for CNN input (add channel dimension)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2],1)  # (samples, n_mfcc, time_frames, 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

print("Fitted data for CNN model SUCCESS")

# Initialize the CNN model
model = Sequential()

print("Building CNN model...")
# Add convolutional layers
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(40, 256, 1)))
# Explanation:
# - 32 filters (patterns the network learns)
# - kernel_size=(3,3): Each filter is a 3x3 sliding window.
# - activation='relu': Introduces non-linearity, making the model learn complex patterns.
# - input_shape=(40, 862, 1): Input dimensions—40 MFCCs, 256 time frames, 1 channel.

model.add(MaxPooling2D(pool_size=(2, 2)))
# Explanation:
# - Reduces dimensionality by taking the max value in 2x2 regions.
# - Makes the model computationally efficient and reduces overfitting.

# Add more convolutional and pooling layers
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the 2D outputs into a 1D vector
model.add(Flatten())

# Add dense (fully connected) layers
model.add(Dense(128, activation='relu'))
# Explanation:
# - Fully connected layer with 128 neurons to learn high-level features.
model.add(Dropout(0.5))
# Explanation:
# - Randomly "drops" 50% of neurons during training to prevent overfitting.

# Output layer
model.add(Dense(y_train.shape[1], activation='softmax'))
# Explanation:
# - Output layer with neurons equal to the number of classes (e.g., emotions).
# - Softmax converts output to probabilities.

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Explanation:
# - optimizer='adam': Adaptive optimization algorithm for faster convergence.
# - loss='categorical_crossentropy': Used for multi-class classification problems.
# - metrics=['accuracy']: Tracks accuracy during training.

print("CNN model build SUCCESS")

print("Training model...")
# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,  # Number of passes through the dataset
    batch_size=32,  # Number of samples per gradient update
    verbose=1  # Displays progress
)
print("Trained model SUCCESS")


model.save('cnn_model40.h5') 
print("Model saved as cnn_model40.h5")

# Evaluate the model on test data
print("Evaluating model...")

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


