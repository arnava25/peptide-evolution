import pandas as pd
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
import os

# === Settings ===
MAX_LEN = 50
AMINO_ACIDS = list('ACDEFGHIKLMNPQRSTVWY')
aa_to_int = {aa: idx + 1 for idx, aa in enumerate(AMINO_ACIDS)}

def encode_sequence(seq, max_len=MAX_LEN):
    if not isinstance(seq, str): return np.zeros(max_len)
    encoded = [aa_to_int.get(aa, 0) for aa in seq[:max_len]]
    return np.pad(encoded, (0, max_len - len(encoded)))

def generate_fake_peptide(length):
    """Generates a random string of amino acids (Maximum Entropy / Alien)."""
    return ''.join(random.choices(AMINO_ACIDS, k=length))

# === 1. Load Real Data (Nature) ===
print("🧬 Loading Natural Sequences...")
# Re-using your AMP dataset, but ignoring the labels. 
# We assume everything in the DB is a "Real" peptide, regardless of activity.
df_real = pd.read_csv('data/model_trainers/amp_dataset.csv')
real_seqs = df_real['Sequence'].dropna().unique().tolist()
real_seqs = [s for s in real_seqs if 10 <= len(s) <= 50] # Filter for realistic lengths

# === 2. Generate Fake Data (Synthetic/Alien) ===
print("👽 Generating Synthetic 'Alien' Sequences...")
fake_seqs = [generate_fake_peptide(random.randint(10, 50)) for _ in range(len(real_seqs))]

# === 3. Combine ===
# Label 1 = Natural / Real
# Label 0 = Synthetic / Alien
X_str = real_seqs + fake_seqs
y = np.array([1] * len(real_seqs) + [0] * len(fake_seqs))

X = np.array([encode_sequence(s) for s in X_str])

# Shuffle
indices = np.arange(len(y))
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

print(f"📊 Training Data: {len(y)} samples (Balanced 50/50)")

# === 4. Build the Discriminator ===
# We use a distinct architecture (LSTM) so it learns different features than your CNNs
model = models.Sequential([
    layers.Embedding(input_dim=22, output_dim=32, mask_zero=True),
    layers.LSTM(64, return_sequences=False), # LSTM is great for "grammar"
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# === 5. Train ===
os.makedirs("models", exist_ok=True)
checkpoint = callbacks.ModelCheckpoint('models/naturalness_discriminator.keras', 
                                     save_best_only=True, monitor='val_loss')
earlystop = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    X, y,
    epochs=10, # Fast training
    batch_size=64,
    validation_split=0.2,
    callbacks=[checkpoint, earlystop]
)

print("✅ 'Turing Test' Discriminator trained and saved to models/naturalness_discriminator.keras")
