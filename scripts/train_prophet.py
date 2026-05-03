import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import glob
import os

# ===============================
# SETTINGS
# ===============================

AMINO_ACIDS = list('ACDEFGHIKLMNPQRSTVWY')
aa_to_idx = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

MAX_LEN = 50          # model max length (peptides will be padded / truncated)
MIN_TRAIN_SAMPLES = 500   # require at least this many examples to train


# ===============================
# ENCODING
# ===============================

def encode_seq(seq: str) -> np.ndarray:
    """
    One-hot encode a peptide sequence into shape (MAX_LEN, 20),
    padded with zeros. Extra residues are truncated.
    """
    x = np.zeros((MAX_LEN, len(AMINO_ACIDS)), dtype="float32")
    for i, aa in enumerate(seq[:MAX_LEN]):
        if aa in aa_to_idx:
            x[i, aa_to_idx[aa]] = 1.0
    return x


# ===============================
# DATA LOADING
# ===============================

def load_reasoning_data():
    """
    Load ALL agent_reasoning CSVs and extract:
        X      : one-hot parent sequence
        y_pos  : one-hot mutated position
        y_aa   : one-hot mutated amino acid
        w      : sample weights based on Reason_Score

    Only uses:
        - rows with Reason_Score > 0  (the agent judged this mutation as beneficial)
        - rows passing a CONSENSUS filter: Cand_AMP > 0.55 AND Cand_Safety > 0.55
          so Prophet learns only from moves that both internal models agreed were good
        - single point mutations (no indels)

    NOTE: Column is Reason_Score (not Total_Score — that was a legacy bug).
    Uses ALL available run logs, not just the latest, to maximise training data.
    """
    # Collect ALL reasoning logs across all runs
    list_of_files = glob.glob("data/agent_reasoning_*.csv")
    if not list_of_files:
        raise FileNotFoundError("No agent_reasoning logs found in data/")

    print(f"📖 Found {len(list_of_files)} reasoning log(s). Loading all...")
    dfs = []
    for f in list_of_files:
        try:
            dfs.append(pd.read_csv(f, on_bad_lines="skip"))
        except Exception as e:
            print(f"  ⚠️ Skipping {f}: {e}")
    if not dfs:
        raise RuntimeError("Could not load any reasoning log.")
    df = pd.concat(dfs, ignore_index=True)

    # Normalise column name: support both legacy Total_Score and current Reason_Score
    if "Reason_Score" in df.columns:
        df = df.rename(columns={"Reason_Score": "Total_Score"})
    elif "Total_Score" not in df.columns:
        raise ValueError("Reasoning log has neither 'Reason_Score' nor 'Total_Score' column.")

    # Basic cleaning
    df = df.dropna(subset=["Parent", "Child", "Total_Score"])
    df["Total_Score"] = pd.to_numeric(df["Total_Score"], errors="coerce")
    df = df.dropna(subset=["Total_Score"])
    print(f"   Total rows loaded: {len(df)}")

    # Keep only "good" moves: positive agent score
    successes = df[df["Total_Score"] > 0.0].copy()
    print(f"✅ Positive-score mutations: {len(successes)} / {len(df)}")

    # === CONSENSUS FILTER ===
    # Only learn from moves where BOTH AMP and Safety scores are above threshold.
    # This prevents Prophet from learning to game one metric at the expense of another.
    AMP_THRESHOLD    = 0.55
    SAFETY_THRESHOLD = 0.55

    if "Cand_AMP" in successes.columns and "Cand_Safety" in successes.columns:
        successes["Cand_AMP"]    = pd.to_numeric(successes["Cand_AMP"],    errors="coerce")
        successes["Cand_Safety"] = pd.to_numeric(successes["Cand_Safety"], errors="coerce")
        before = len(successes)
        successes = successes[
            (successes["Cand_AMP"]    >= AMP_THRESHOLD) &
            (successes["Cand_Safety"] >= SAFETY_THRESHOLD)
        ].copy()
        print(f"🔍 Consensus filter (AMP≥{AMP_THRESHOLD}, Safety≥{SAFETY_THRESHOLD}): "
              f"{len(successes)} / {before} passed")
    else:
        print("⚠️  Cand_AMP / Cand_Safety columns not found — skipping consensus filter.")

    X = []
    y_pos = []
    y_aa = []
    weights = []

    for _, row in successes.iterrows():
        parent = str(row["Parent"])
        child = str(row["Child"])

        # Skip indels (only allow same-length sequences)
        if len(parent) != len(child):
            continue

        # Identify positions where parent and child differ
        diffs = [i for i in range(len(parent)) if parent[i] != child[i]]
        if len(diffs) != 1:
            # Only learn from single point mutations for now
            continue

        idx = diffs[0]
        if idx >= MAX_LEN:
            # Out-of-range for our fixed model length
            continue

        new_aa = child[idx]
        if new_aa not in aa_to_idx:
            continue

        # Encode parent
        X.append(encode_seq(parent))

        # Target 1: position
        pos_vec = np.zeros(MAX_LEN, dtype="float32")
        pos_vec[idx] = 1.0
        y_pos.append(pos_vec)

        # Target 2: amino acid
        aa_vec = np.zeros(len(AMINO_ACIDS), dtype="float32")
        aa_vec[aa_to_idx[new_aa]] = 1.0
        y_aa.append(aa_vec)

        # Sample weight: how good was this move?
        weights.append(float(row["Total_Score"]))

    if not X:
        print("❌ No usable single-point successful mutations found.")
        return None, None, None, None

    X = np.array(X, dtype="float32")
    y_pos = np.array(y_pos, dtype="float32")
    y_aa = np.array(y_aa, dtype="float32")
    weights = np.array(weights, dtype="float32")

    # Normalize / clip weights so training is stable
    # - clip extreme outliers
    # - rescale so mean weight ≈ 1.0
    weights = np.clip(weights, 1e-4, np.percentile(weights, 95))
    weights /= np.mean(weights)

    print(f"📊 Final dataset: {len(X)} samples")
    print(f"   Mean weight: {weights.mean():.3f} | "
          f"Min weight: {weights.min():.3f} | Max weight: {weights.max():.3f}")

    return X, y_pos, y_aa, weights


# ===============================
# MODEL
# ===============================

def build_prophet() -> keras.Model:
    """
    Build the Prophet model:
        Input  : (MAX_LEN, 20) one-hot sequence
        Shared : Masking -> LSTM(64) -> Dense(64, relu + dropout)
        Heads  : position softmax, amino-acid softmax
    """
    inp = layers.Input(shape=(MAX_LEN, len(AMINO_ACIDS)), name="seq")

    # Ignore padded timesteps (all zeros)
    x = layers.Masking(mask_value=0.0)(inp)

    # Sequence processor
    x = layers.LSTM(64, return_sequences=False)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    # Head 1: where to mutate?
    out_pos = layers.Dense(MAX_LEN, activation="softmax", name="position")(x)

    # Head 2: what AA to insert?
    out_aa = layers.Dense(len(AMINO_ACIDS), activation="softmax", name="aa")(x)


    model = keras.Model(inputs=inp, outputs=[out_pos, out_aa])

    # Use positional losses/metrics to avoid name–dict weirdness in Keras 3
    model.compile(
        optimizer="adam",
        loss=["categorical_crossentropy", "categorical_crossentropy"],
        metrics=["accuracy", "accuracy"],
    )

    model.summary()
    return model


# ===============================
# MAIN
# ===============================

if __name__ == "__main__":
    X, y_pos, y_aa, weights = load_reasoning_data()

    if X is None or len(X) < MIN_TRAIN_SAMPLES:
        print(f"❌ Not enough training data "
              f"({0 if X is None else len(X)} < {MIN_TRAIN_SAMPLES}). "
              "Did the last run log reasons correctly?")
        raise SystemExit(0)

    print("🔮 Training The Prophet (stronger version)...")

    model = build_prophet()

    # Early stopping to avoid overfitting
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        )
    ]

    history = model.fit(
        X,
        [y_pos, y_aa],              # match outputs [out_pos, out_aa]
        sample_weight=[weights, weights],  # one weight vector per output
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1,
    )

    
    # Save model
    os.makedirs("models", exist_ok=True)
    out_path = "models/prophet_model.keras"
    model.save(out_path)
    print(f"💾 Prophet model saved to {out_path}")
    print("🚀 Ready for Run 5: Stronger AI-Guided Evolution.")

