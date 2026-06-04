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
    Load agent_reasoning data with contrastive training:
    - Accepted mutations: learn TO mutate this position/AA (positive weight)
    - Rejected mutations: learn to AVOID this position/AA (negative weight via label smoothing)
    Uses combined 25mer reasoning log.
    """
    candidate_files = [
        "data/agent_reasoning_25mer_combined.csv",
    ]
    # fallback to any reasoning log in data/
    list_of_files = [f for f in candidate_files if os.path.exists(f)]
    if not list_of_files:
        list_of_files = glob.glob("data/agent_reasoning_*.csv")
    if not list_of_files:
        raise FileNotFoundError("No agent_reasoning logs found.")

    print(f"📖 Loading: {list_of_files}")
    dfs = []
    for f in list_of_files:
        try:
            dfs.append(pd.read_csv(f, on_bad_lines="skip"))
        except Exception as e:
            print(f"  ⚠️ Skipping {f}: {e}")
    if not dfs:
        raise RuntimeError("Could not load any reasoning log.")
    df = pd.concat(dfs, ignore_index=True)

    # Normalise score column name
    if "Reason_Score" in df.columns:
        df = df.rename(columns={"Reason_Score": "Total_Score"})
    elif "Total_Score" not in df.columns:
        raise ValueError("No Reason_Score or Total_Score column found.")

    df = df.dropna(subset=["Parent", "Child", "Total_Score"])
    df = df[df['Parent'].str.len() == 25].copy()
    df = df[df['Child'].str.len() == 25].copy()
    print(f"   After 25mer filter: {len(df)}")

    df["Total_Score"] = pd.to_numeric(df["Total_Score"], errors="coerce")
    df = df.dropna(subset=["Total_Score"])

    # Tag accepted vs rejected
    df["is_rejected"] = df["Source"].str.contains("rejected", na=False)
    accepted = df[~df["is_rejected"]].copy()
    rejected = df[df["is_rejected"]].copy()
    print(f"✅ Accepted: {len(accepted)} | ❌ Rejected: {len(rejected)}")

    # Consensus filter on accepted only
    AMP_THRESHOLD    = 0.70
    SAFETY_THRESHOLD = 0.65
    if "Cand_AMP" in accepted.columns and "Cand_Safety" in accepted.columns:
        accepted["Cand_AMP"]    = pd.to_numeric(accepted["Cand_AMP"],    errors="coerce")
        accepted["Cand_Safety"] = pd.to_numeric(accepted["Cand_Safety"], errors="coerce")
        before = len(accepted)
        accepted = accepted[
            (accepted["Cand_AMP"]    >= AMP_THRESHOLD) &
            (accepted["Cand_Safety"] >= SAFETY_THRESHOLD)
        ].copy()
        print(f"🔍 Consensus filter: {len(accepted)} / {before} accepted passed")

    # Downsample rejected to 2x accepted to keep balance
    n_rej = min(len(rejected), len(accepted) * 2)
    rejected = rejected.sample(n=n_rej, random_state=42).copy()
    print(f"⚖️  Using {len(accepted)} accepted + {len(rejected)} rejected")

    X, y_pos, y_aa, weights = [], [], [], []

    for _, row in accepted.iterrows():
        parent, child = str(row["Parent"]), str(row["Child"])
        if len(parent) != len(child): continue
        diffs = [i for i in range(len(parent)) if parent[i] != child[i]]
        if len(diffs) != 1: continue
        idx = diffs[0]
        if idx >= MAX_LEN: continue
        new_aa = child[idx]
        if new_aa not in aa_to_idx: continue

        X.append(encode_seq(parent))
        pos_vec = np.zeros(MAX_LEN, dtype="float32")
        pos_vec[idx] = 1.0
        y_pos.append(pos_vec)
        aa_vec = np.zeros(len(AMINO_ACIDS), dtype="float32")
        aa_vec[aa_to_idx[new_aa]] = 1.0
        y_aa.append(aa_vec)
        weights.append(max(float(row["Total_Score"]), 0.01))

    for _, row in rejected.iterrows():
        parent, child = str(row["Parent"]), str(row["Child"])
        if len(parent) != len(child): continue
        diffs = [i for i in range(len(parent)) if parent[i] != child[i]]
        if len(diffs) != 1: continue
        idx = diffs[0]
        if idx >= MAX_LEN: continue
        new_aa = child[idx]
        if new_aa not in aa_to_idx: continue

        X.append(encode_seq(parent))
        # For rejected: teach prophet to mutate a DIFFERENT position
        # Use a uniform distribution with the rejected position suppressed
        pos_vec = np.ones(MAX_LEN, dtype="float32")
        pos_vec[idx] = 0.0  # zero out rejected position
        pos_vec /= pos_vec.sum()
        y_pos.append(pos_vec)
        # Uniform over AAs except the rejected one
        aa_vec = np.ones(len(AMINO_ACIDS), dtype="float32")
        aa_vec[aa_to_idx[new_aa]] = 0.0
        aa_vec /= aa_vec.sum()
        y_aa.append(aa_vec)
        weights.append(0.3)  # small positive weight, not negative
    
    if not X:
        print("❌ No usable training samples found.")
        return None, None, None, None

    X       = np.array(X,       dtype="float32")
    y_pos   = np.array(y_pos,   dtype="float32")
    y_aa    = np.array(y_aa,    dtype="float32")
    weights = np.array(weights, dtype="float32")

    # Clip and normalize — keep sign for contrastive signal
    weights = np.clip(weights, 1e-4, np.percentile(weights, 95))
    weights /= np.mean(weights)

    print(f"📊 Final dataset: {len(X)} samples")
    print(f"   Mean weight: {weights.mean():.3f} | Min: {weights.min():.3f} | Max: {weights.max():.3f}")

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

    print(f"📊 Dataset stats:")
    print(f"   Unique parent sequences: {len(set([str(s.tolist()) for s in X]))}")
    print(f"   Position distribution (top 5): {dict(sorted(zip(*np.unique(y_pos.argmax(axis=1), return_counts=True)), key=lambda x: -x[1])[:5])}")
    print(f"   AA distribution (top 5): {dict(sorted(zip(*np.unique(y_aa.argmax(axis=1), return_counts=True)), key=lambda x: -x[1])[:5])}")
    print("🔮 Training The Prophet (stronger version)...")

    model = build_prophet()

    # Early stopping to avoid overfitting
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            verbose=1,
            min_lr=1e-6,
        ),
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

