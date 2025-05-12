import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
import os

# === Callbacks ===
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Load dataset
df = pd.read_csv('data/model_trainers/toxicity_dataset_balanced.csv')

amino_acids = list('ACDEFGHIKLMNPQRSTVWY')
aa_to_int = {aa: idx+1 for idx, aa in enumerate(amino_acids)}

def encode_sequence(seq, max_len=50):
    if not isinstance(seq, str):
        seq = ''
    encoded = [aa_to_int.get(aa, 0) for aa in seq[:max_len]]
    return np.pad(encoded, (0, max_len - len(encoded)))

X = np.array([encode_sequence(seq) for seq in df['Sequence']])
y = df['Label'].values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = models.Sequential([
    layers.Embedding(input_dim=21, output_dim=64),
    layers.Conv1D(128, 5, activation='relu'),
    layers.Conv1D(128, 5, activation='relu'),
    layers.GlobalMaxPooling1D(),
    layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    layers.Dropout(0.4),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

checkpoint = callbacks.ModelCheckpoint('models/toxicity_cnn_model.keras', save_best_only=True, monitor='val_loss')
earlystop = callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

model.fit(X_train, y_train, epochs=35, batch_size=64, validation_data=(X_val, y_val),
          callbacks=[checkpoint, earlystop, reduce_lr])

print("✅ Toxicity model upgraded and saved!")
