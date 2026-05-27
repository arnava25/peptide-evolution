import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import os

os.makedirs('logs', exist_ok=True)

df = pd.read_csv('data/model_trainers/mic_ecoli_dataset.csv')
print(f"Dataset: {len(df)} peptides")

# Log transform MIC
df['log_MIC'] = np.log10(df['MIC'])

amino_acids = list('ACDEFGHIKLMNPQRSTVWY')
aa_to_idx = {aa: idx+1 for idx, aa in enumerate(amino_acids)}
max_len = 50

def encode_int(seq, max_len=50):
    x = np.zeros(max_len, dtype=np.int32)
    for i, aa in enumerate(seq[:max_len].upper()):
        idx = aa_to_idx.get(aa)
        if idx is not None:
            x[i] = idx
    return x

X = np.array([encode_int(s) for s in df['SEQUENCE']])
y = df['log_MIC'].values.astype('float32')

# Train/val split
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)
print(f"Train: {len(X_train)}, Val: {len(X_val)}")

# Build CNN regression model
inp = layers.Input(shape=(max_len,), dtype='int32')
x = layers.Embedding(input_dim=21, output_dim=16, mask_zero=False)(inp)
x = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
x = layers.Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(64, activation='relu')(x)
out = layers.Dense(1, activation='linear')(x)

model = keras.Model(inputs=inp, outputs=out)
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse', metrics=['mae'])
model.summary()

checkpoint = callbacks.ModelCheckpoint('models/mic_ecoli_model.keras', save_best_only=True, monitor='val_mae')
earlystop = callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor='val_mae')
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_mae', factor=0.5, patience=4)

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=64,
    callbacks=[checkpoint, earlystop, reduce_lr]
)

val_preds = model.predict(X_val).flatten()
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_val, val_preds)
print(f"\nFinal val MAE (log10 MIC): {mae:.4f}")
print(f"That means predictions are within {10**mae:.2f}x of true MIC on average")
print("✅ MIC model saved to models/mic_ecoli_model.keras")
