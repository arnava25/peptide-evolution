import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from collections import Counter
import os

# === Callbacks ===
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# === Load AMP dataset ===
df = pd.read_csv('data/model_trainers/amp_dataset.csv')

assert 'Sequence' in df.columns and 'Label' in df.columns, "CSV must have 'Sequence' and 'Label' columns."

# Remove bad rows
df = df[df['Sequence'].apply(lambda x: isinstance(x, str) and len(x) > 0)]
df = df[df['Label'].isin([0, 1])]

print(f"📊 Dataset loaded: {len(df)} sequences")
print(df['Label'].value_counts())

# === Encoding ===
amino_acids = list('ACDEFGHIKLMNPQRSTVWY')
aa_to_int = {aa: idx + 1 for idx, aa in enumerate(amino_acids)}

def encode_sequence(seq, max_len=50):
    encoded = [aa_to_int.get(aa, 0) for aa in seq[:max_len]]
    return np.pad(encoded, (0, max_len - len(encoded)))

X = np.array([encode_sequence(seq) for seq in df['Sequence']])
y = df['Label'].values

# === Train/val split ===
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === Model ===
model = models.Sequential([
    layers.Embedding(input_dim=21, output_dim=32, input_length=50),
    layers.Conv1D(64, 5, activation='relu'),
    layers.Conv1D(64, 5, activation='relu'),
    layers.GlobalAveragePooling1D(),
    layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# === Class weights ===
counts = Counter(y_train)
total = sum(counts.values())
class_weight = {
    0: total / (2.0 * counts[0]),
    1: total / (2.0 * counts[1])
}
print(f"⚖️  Class weights: {class_weight}")

checkpoint = callbacks.ModelCheckpoint('models/amp_model.keras', save_best_only=True, monitor='val_loss')
earlystop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1, min_lr=1e-6)
csv_logger = callbacks.CSVLogger('logs/amp_training_log.csv', append=False)

# === Train ===
history = model.fit(
    X_train, y_train,
    epochs=35,
    batch_size=64,
    validation_data=(X_val, y_val),
    class_weight=class_weight,
    callbacks=[checkpoint, earlystop, reduce_lr, csv_logger]
)

# === Save architecture ===
with open("models/amp_model_summary.txt", "w") as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))

print("✅ AMP model upgraded and saved!")
